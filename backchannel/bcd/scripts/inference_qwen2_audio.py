"""
Qwen2-Audio 기반 백채널 검출 추론 스크립트

기존 매니페스트에서 샘플을 읽고, 믹싱 오디오 + 프롬프트를 Qwen2-Audio에
입력하여 백채널 클래스를 생성형으로 분류한다.

사용법:
    # 단일 GPU
    python scripts/inference_qwen2_audio.py --config configs/qwen2_audio.yaml
    python scripts/inference_qwen2_audio.py --config configs/qwen2_audio.yaml --split test --max_samples 10

    # 다중 GPU (torchrun) — 각 rank가 samples[rank::world_size]만 처리,
    # rank 0이 shard merge 후 evaluation 수행
    torchrun --standalone --nproc_per_node=4 scripts/inference_qwen2_audio.py \
        --config configs/qwen2_audio.yaml --split test
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bc_detection.qwen2.adapter import Qwen2AudioAdapter
from bc_detection.qwen2.dataset import Qwen2AudioDataset
from bc_detection.qwen2.evaluator import Qwen2AudioEvaluator
from bc_detection.qwen2.fewshot import FewShotConfig, FewShotSelector
from bc_detection.qwen2.label_parser import QwenLabelParser
from bc_detection.qwen2.metadata import get_git_info
from bc_detection.qwen2.prompt_builder import QwenPromptBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """YAML 설정 파일을 로드한다."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dist_info():
    """torchrun이 부여한 RANK/WORLD_SIZE를 읽는다. 단일 프로세스면 (0, 1)."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def wait_for_shards(shard_paths, timeout_sec=86400, poll_sec=5):
    """모든 shard sentinel 파일이 생길 때까지 대기 (file-based barrier)."""
    deadline = time.time() + timeout_sec
    pending = [Path(p) for p in shard_paths]
    while pending:
        pending = [p for p in pending if not p.exists()]
        if not pending:
            return
        if time.time() > deadline:
            raise TimeoutError(f"shard 대기 timeout. 누락: {pending}")
        time.sleep(poll_sec)


def extract_audio_paths_from_prompt(prompt, fallback_paths):
    """chat template prompt에 포함된 audio URL을 순서대로 추출한다."""
    if not isinstance(prompt, list):
        return fallback_paths

    audio_paths = []
    for message in prompt:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "audio":
                    audio_paths.append(item["audio_url"])

    return audio_paths or fallback_paths


def resolve_lora_weights_path(config: dict):
    """inference에서 사용할 LoRA weights path를 결정한다."""
    qwen2_cfg = config["qwen2_audio"]
    lora_cfg = qwen2_cfg.get("lora", {})
    if not lora_cfg.get("enabled"):
        return None

    weights_path = lora_cfg.get("weights_path")
    if weights_path:
        return weights_path

    train_cfg = config.get("training", {})
    output_dir = train_cfg.get("output_dir")
    if output_dir:
        final_path = Path(output_dir) / "final"
        if final_path.exists():
            logger.warning(
                "lora.weights_path가 비어 있어 training.output_dir/final을 사용합니다: %s",
                final_path,
            )
            return str(final_path)

    logger.warning("lora.enabled=true이지만 weights_path를 찾지 못했습니다. LoRA 없이 추론합니다.")
    return None


def build_fewshot_selector(config: dict, data_path: Path):
    """dynamic few-shot selector를 구성한다. 비활성 시 (config, None)을 반환한다."""
    fewshot_config = FewShotConfig.from_dict(
        config.get("qwen2_audio", {}).get("dynamic_fewshot")
    )
    if not fewshot_config.enabled:
        return fewshot_config, None

    if config.get("qwen2_audio", {}).get("fewshot", 0):
        raise ValueError("qwen2_audio.fewshot과 dynamic_fewshot은 동시에 사용할 수 없습니다")

    data_cfg = config["data"]
    if data_cfg.get("audio_input_mode", "mixed") != "separated":
        raise ValueError("dynamic_fewshot 1차 구현은 separated audio_input_mode만 지원합니다")

    support_dataset = Qwen2AudioDataset(
        manifest_path=data_path / data_cfg.get("train_manifest", "splits/train"),
        audio_root=data_path,
        label_mapping=data_cfg.get("label_mapping"),
        mixing_audio_dir=data_cfg.get("mixing_audio_dir", "mixing_audios"),
        speaker_audio_dir=data_cfg.get("speaker_audio_dir", "speaker_audios"),
        listener_audio_dir=data_cfg.get("listener_audio_dir", "listener_audios"),
        audio_input_mode=data_cfg.get("audio_input_mode", "mixed"),
        validate_audio=False,
    )
    logger.info("Few-shot support pool 로드: train split, %d개", len(support_dataset))
    return fewshot_config, FewShotSelector(
        support_samples=support_dataset.samples,
        config=fewshot_config,
    )


def run_inference(
    config: dict,
    split: str = "test",
    max_samples: int = None,
    config_path: str = None,
):
    """
    Qwen2-Audio 추론 파이프라인을 실행한다.

    1단계 목표:
    - 매니페스트에서 샘플 목록 읽기
    - mixing_audios/에서 오디오 로드
    - Qwen2-Audio generation
    - raw output + parsed label 저장
    """
    qwen2_cfg = config["qwen2_audio"]
    data_cfg = config["data"]
    infer_cfg = config.get("inference", {})

    rank, world_size = get_dist_info()
    is_distributed = world_size > 1
    is_main = rank == 0
    if is_distributed:
        logger.info(f"distributed inference: rank={rank}/{world_size}")

    # === 1. 데이터셋 로드 ===
    data_path = Path(data_cfg["data_path"])
    manifest_map = {
        "train": data_cfg.get("train_manifest", "splits/train"),
        "valid": data_cfg.get("val_manifest", "splits/valid"),
        "test": data_cfg.get("test_manifest", "splits/test"),
    }

    if split not in manifest_map:
        raise ValueError(f"split은 {list(manifest_map.keys())} 중 하나여야 합니다: {split}")

    manifest_path = data_path / manifest_map[split]

    dataset = Qwen2AudioDataset(
        manifest_path=manifest_path,
        audio_root=data_path,
        label_mapping=data_cfg.get("label_mapping"),
        mixing_audio_dir=data_cfg.get("mixing_audio_dir", "mixing_audios"),
        speaker_audio_dir=data_cfg.get("speaker_audio_dir", "speaker_audios"),
        listener_audio_dir=data_cfg.get("listener_audio_dir", "listener_audios"),
        audio_input_mode=data_cfg.get("audio_input_mode", "mixed"),
        validate_audio=data_cfg.get("validate_audio", False),
        crop_speaker_sec=data_cfg.get("crop_speaker_sec"),
        crop_listener_sec=data_cfg.get("crop_listener_sec"),
        silence_speaker=data_cfg.get("silence_speaker", False),
        silence_listener=data_cfg.get("silence_listener", False),
    )

    logger.info(f"데이터셋 로드: {split} split, {len(dataset)}개 샘플")
    logger.info(f"클래스 분포: {dataset.get_class_distribution()}")

    fewshot_config, fewshot_selector = build_fewshot_selector(config, data_path)

    # max_samples 적용
    total_samples = len(dataset)
    if max_samples is not None:
        total_samples = min(max_samples, total_samples)
        logger.info(f"max_samples 적용: {total_samples}개 샘플만 사용")

    # === 2. 모델 로드 ===
    adapter = Qwen2AudioAdapter(
        model_type=qwen2_cfg.get("model_type", "instruct"),
        model_id=qwen2_cfg.get("model_id"),
        device_map=qwen2_cfg.get("device_map", "auto"),
        torch_dtype=qwen2_cfg.get("torch_dtype", "float16"),
        max_new_tokens=qwen2_cfg.get("max_new_tokens", 32),
        do_sample=qwen2_cfg.get("do_sample", False),
        attn_implementation=qwen2_cfg.get("attn_implementation", "sdpa"),
    )
    # LoRA 로딩: (a) MLflow Registry pyfunc(@alias) 또는 (b) 로컬 weights_path (M10)
    lora_cfg = qwen2_cfg.get("lora", {})
    registry_uri = lora_cfg.get("registry_uri")  # 예: models:/qwen2audio-bc-swbd@candidate
    if registry_uri:
        import mlflow

        registry_tracking_uri = lora_cfg.get("registry_tracking_uri")
        if registry_tracking_uri:
            mlflow.set_tracking_uri(registry_tracking_uri)
        logger.info("MLflow Registry에서 LoRA 로드: %s", registry_uri)
        # pyfunc는 loader 전용 — unwrap 해 결합된 Qwen2AudioAdapter(base HF캐시+adapter)를 회수한다.
        # (base weight는 artifact에 없고 추론 노드 HF 캐시에서 로드됨; §14)
        adapter = mlflow.pyfunc.load_model(registry_uri).unwrap_python_model().adapter
    else:
        adapter.load_model()
        lora_weights_path = resolve_lora_weights_path(config)
        if lora_weights_path:
            adapter.load_lora_weights(lora_weights_path)

    # === 3. 프롬프트 빌더 ===
    prompt_builder = QwenPromptBuilder(
        use_chat_template=qwen2_cfg.get("use_chat_template", True),
        use_speaker_text=qwen2_cfg.get("use_speaker_text", False),
        use_listener_text=qwen2_cfg.get("use_listener_text", False),
        model_type=qwen2_cfg.get("model_type", "instruct"),
        prompt_version=qwen2_cfg.get("prompt_version", "v1"),
        prompt_templates=qwen2_cfg.get("prompt_templates"),
        fewshot=qwen2_cfg.get("fewshot", 0),
        audio_input_mode=data_cfg.get("audio_input_mode", "mixed"),
    )

    # === 4. 레이블 파서 ===
    parser_cfg = qwen2_cfg.get("label_parser", {})
    # data.valid_labels (SWBD = 3-class, BCKO = 4-class default) 를 parser regex에 주입
    label_parser = QwenLabelParser(
        fallback_label=parser_cfg.get("fallback_label"),
        valid_labels=data_cfg.get("valid_labels"),
    )

    # === 5. 추론 실행 ===
    audio_input_mode = data_cfg.get("audio_input_mode", "mixed")
    batch_size = data_cfg.get("batch_size", 4)
    if audio_input_mode == "separated" and batch_size != 1:
        logger.info("separated mode smoke 경로는 batch_size=1로 실행합니다")
        batch_size = 1
    use_chat_template = qwen2_cfg.get("use_chat_template", True)

    results = []
    start_time = time.time()

    # rank별 처리할 sample 인덱스 (round-robin shard)
    my_indices = list(range(rank, total_samples, world_size)) if is_distributed else list(range(total_samples))
    my_total = len(my_indices)
    if is_distributed:
        logger.info(f"rank {rank}: {my_total}/{total_samples} samples 할당")

    for batch_start in range(0, my_total, batch_size):
        batch_end = min(batch_start + batch_size, my_total)
        batch_indices = my_indices[batch_start:batch_end]

        # 배치 데이터 준비
        audios = []
        prompts = []
        batch_samples = []
        batch_support_examples = []

        for idx in batch_indices:
            sample = dataset[idx]
            batch_samples.append(sample)

            audio_paths = dataset.get_audio_paths(sample)
            path_to_role = {p: r for p, r in dataset.get_audio_paths_with_roles(sample)}
            support_examples = []
            if fewshot_selector is not None:
                support_examples = fewshot_selector.select(
                    target_sample=sample,
                    target_index=idx,
                    variant_id=0,
                )
            batch_support_examples.append(support_examples)

            # 프롬프트 생성
            prompt = prompt_builder.build(
                audio_paths=audio_paths,
                speaker_text=sample.get("speaker_text"),
                listener_text=sample.get("listener_text"),
                support_examples=support_examples,
            )
            prompt_audio_paths = extract_audio_paths_from_prompt(prompt, audio_paths)
            for audio_path in prompt_audio_paths:
                role = path_to_role.get(audio_path, "mixed")
                audios.append(dataset.load_audio_for_role(audio_path, role))
            prompts.append(prompt)

        # generation 실행 — runtime 오류 발생 시 batch 단위로 skip (해당 sample들은 invalid 처리)
        # 실패 원인 예: chat template에서 audio placeholder 누락 → _merge_input_ids_with_audio_features
        # 의 shape mismatch. Full test 104k 중 매우 드물게 발생하므로 fail-safe로 진행.
        try:
            if audio_input_mode == "separated":
                raw_outputs = [
                    adapter.generate_multi_audio_single(
                        audios=audios,
                        prompt=prompts[0],
                        use_chat_template=use_chat_template,
                    )
                ]
            else:
                raw_outputs = adapter.generate(
                    audios=audios,
                    prompts=prompts,
                    use_chat_template=use_chat_template,
                )
        except (RuntimeError, ValueError) as e:
            failed_ids = [s.get("sample_id") for s in batch_samples]
            logger.warning(f"[rank {rank}] generation 실패 (skip): sample_ids={failed_ids} err={type(e).__name__}: {e}")
            raw_outputs = ["<<GENERATION_ERROR>>"] * len(batch_samples)

        # 레이블 파싱
        for sample, raw_output, support_examples in zip(
            batch_samples,
            raw_outputs,
            batch_support_examples,
        ):
            cleaned, parsed_label, valid = label_parser.parse(raw_output)

            results.append({
                "sample_id": sample["sample_id"],
                "gold_label": sample["label"],
                "raw_output": raw_output,
                "cleaned_output": cleaned,
                "parsed_label": parsed_label,
                "valid_parse": valid,
                "support_sample_ids": [
                    support.get("sample_id") for support in support_examples
                ] if fewshot_selector is not None else [],
                "support_labels": [
                    support.get("label") for support in support_examples
                ] if fewshot_selector is not None else [],
            })

        # 진행 상황 로깅
        processed = batch_end
        elapsed = time.time() - start_time
        speed = processed / elapsed if elapsed > 0 else 0
        prefix = f"[rank {rank}] " if is_distributed else ""
        logger.info(
            f"{prefix}진행: {processed}/{my_total} "
            f"({processed/my_total*100:.1f}%, {speed:.1f} samples/sec)"
        )

    total_time = time.time() - start_time

    # === 6. 결과 저장 ===
    output_dir = Path(infer_cfg.get("output_dir", "results/qwen2_audio"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 샘플 단위 결과 저장 (raw output 포함 — 지침서 15.4)
    save_format = infer_cfg.get("save_format", "jsonl")

    if is_distributed:
        # 각 rank는 본인 shard를 별도 파일로 저장. csv는 분산 모드에서 미지원.
        if save_format != "jsonl":
            raise ValueError("distributed inference는 save_format=jsonl만 지원합니다")
        shard_path = output_dir / f"predictions_{split}.rank{rank}_of{world_size}.{save_format}"
        with open(shard_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # sentinel
        sentinel_path = shard_path.with_suffix(shard_path.suffix + ".done")
        sentinel_path.touch()
        logger.info(f"[rank {rank}] shard 저장: {shard_path} ({len(results)} samples)")

        if not is_main:
            # non-main rank는 여기서 종료. evaluation은 rank 0가 수행.
            logger.info(f"[rank {rank}] shard 완료, 종료")
            return None, None

        # rank 0: 모든 shard sentinel 대기 후 merge
        shard_sentinels = [
            output_dir / f"predictions_{split}.rank{r}_of{world_size}.{save_format}.done"
            for r in range(world_size)
        ]
        logger.info(f"[rank 0] {world_size}개 shard 대기 중...")
        wait_for_shards(shard_sentinels)

        # merge: shard들을 합쳐 최종 predictions 파일 생성
        sample_results_path = output_dir / f"predictions_{split}.{save_format}"
        merged_results = []
        with open(sample_results_path, "w", encoding="utf-8") as out:
            for r in range(world_size):
                shard = output_dir / f"predictions_{split}.rank{r}_of{world_size}.{save_format}"
                with open(shard, "r", encoding="utf-8") as f:
                    for line in f:
                        out.write(line)
                        merged_results.append(json.loads(line))
        logger.info(f"[rank 0] merge 완료: {sample_results_path} ({len(merged_results)} samples)")

        # shard 임시 파일 정리
        for r in range(world_size):
            shard = output_dir / f"predictions_{split}.rank{r}_of{world_size}.{save_format}"
            sentinel = shard.with_suffix(shard.suffix + ".done")
            shard.unlink(missing_ok=True)
            sentinel.unlink(missing_ok=True)

        results = merged_results
    else:
        sample_results_path = output_dir / f"predictions_{split}.{save_format}"
        if save_format == "jsonl":
            with open(sample_results_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        elif save_format == "csv":
            import csv
            with open(sample_results_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

    logger.info(f"샘플 결과 저장: {sample_results_path}")

    # === 7. 평가 (기존 metric 체계 연동) — rank 0만 ===
    # data.valid_labels (SWBD 3-class) override 시 evaluator도 동일 class set 사용
    eval_class_names = data_cfg.get("valid_labels")
    eval_class_names = sorted(eval_class_names) if eval_class_names else None
    evaluator = Qwen2AudioEvaluator(exclude_invalid=True, class_names=eval_class_names)
    evaluation = evaluator.evaluate(results)

    # 메타데이터 추가
    # distributed 모드는 rank 0의 시작~merge 완료까지 wall-clock 사용 (가장 늦은 rank 포함)
    if is_distributed:
        total_time = time.time() - start_time
    evaluation["timestamp"] = datetime.now().isoformat()
    evaluation["split"] = split
    evaluation["config_path"] = config_path
    evaluation["total_time_sec"] = round(total_time, 2)
    evaluation["samples_per_sec"] = (
        round(total_samples / total_time, 2) if total_time > 0 else 0
    )
    evaluation["world_size"] = world_size
    evaluation["gold_label_distribution"] = dataset.get_class_distribution()
    evaluation.update(adapter.get_metadata())
    evaluation.update(prompt_builder.get_metadata())
    evaluation.update(get_git_info(Path(__file__).resolve().parent.parent))
    evaluation["lora_enabled"] = bool(qwen2_cfg.get("lora", {}).get("enabled"))
    evaluation["lora_weights_path"] = lora_weights_path
    evaluation["dynamic_fewshot"] = fewshot_config.to_dict()

    # 평가 결과 저장 (JSON + confusion matrix PNG + classification report TXT)
    evaluator.save_evaluation(evaluation, output_dir, split)

    # 콘솔 요약 출력
    evaluator.print_summary(evaluation, split)
    logger.info(f"총 시간: {total_time:.1f}초 ({evaluation['samples_per_sec']} samples/sec)")
    logger.info(f"결과 디렉토리: {output_dir}")

    return results, evaluation


def main():
    parser = argparse.ArgumentParser(description="Qwen2-Audio 백채널 검출 추론")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen2_audio.yaml",
        help="설정 파일 경로",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="평가할 데이터 split",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="최대 샘플 수 (디버깅용)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # CLI에서 max_samples 지정 시 config보다 우선
    max_samples = args.max_samples
    if max_samples is None:
        max_samples = config.get("inference", {}).get("max_samples")

    run_inference(
        config,
        split=args.split,
        max_samples=max_samples,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()

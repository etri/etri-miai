"""
Qwen2-Audio LoRA 학습 스크립트

Qwen2-Audio 모델에 LoRA를 적용하여 백채널 분류 태스크로 fine-tuning한다.
학습 방식: 오디오 + 프롬프트 → 레이블 텍스트 생성 (generative classification)

사용법:
    python scripts/train_qwen2_audio.py --config configs/qwen2_audio.yaml
    python scripts/train_qwen2_audio.py --config configs/qwen2_audio.yaml --max_train_samples 100
"""

import argparse
import copy
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bc_detection.qwen2.adapter import Qwen2AudioAdapter
from bc_detection.qwen2.aim_logging import build_aim_callback
from bc_detection.qwen2.dataset import Qwen2AudioDataset
from bc_detection.qwen2.evaluator import Qwen2AudioEvaluator
from bc_detection.qwen2.fewshot import FewShotConfig, FewShotSelector
from bc_detection.qwen2.label_parser import QwenLabelParser
from bc_detection.qwen2.lora_config import QwenLoraConfigBuilder
from bc_detection.qwen2.metadata import get_git_info, save_resolved_config
from bc_detection.qwen2.prompt_builder import QwenPromptBuilder
from bc_detection.qwen2.trainer import setup_training, save_lora_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """YAML 설정 파일을 로드한다."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def build_fewshot_selector(config: dict, support_dataset):
    """dynamic few-shot selector를 구성한다. 비활성 시 (config, None)을 반환한다."""
    fewshot_config = FewShotConfig.from_dict(
        config.get("qwen2_audio", {}).get("dynamic_fewshot")
    )
    if not fewshot_config.enabled:
        return fewshot_config, None

    if config.get("qwen2_audio", {}).get("fewshot", 0):
        raise ValueError("qwen2_audio.fewshot과 dynamic_fewshot은 동시에 사용할 수 없습니다")

    if config.get("data", {}).get("audio_input_mode", "mixed") != "separated":
        raise ValueError("dynamic_fewshot 1차 구현은 separated audio_input_mode만 지원합니다")

    return fewshot_config, FewShotSelector(
        support_samples=support_dataset.samples,
        config=fewshot_config,
    )


def run_training(
    config: dict,
    max_train_samples: int = None,
    max_val_samples: int = None,
    config_path: str = None,
    resume_from_checkpoint: str = None,
):
    """
    Qwen2-Audio LoRA 학습 파이프라인을 실행한다.

    1. 데이터셋 로드 (train + valid)
    2. 모델 로드 + LoRA 적용
    3. 학습 실행
    4. LoRA 가중치 저장
    5. (선택) 학습 후 평가
    """
    qwen2_cfg = config["qwen2_audio"]
    data_cfg = config["data"]
    train_cfg = config.get("training", {})
    lora_cfg = qwen2_cfg.get("lora", {})

    # === 1. 데이터셋 로드 ===
    data_path = Path(data_cfg["data_path"])

    common_dataset_kwargs = dict(
        audio_root=data_path,
        label_mapping=data_cfg.get("label_mapping"),
        mixing_audio_dir=data_cfg.get("mixing_audio_dir", "mixing_audios"),
        speaker_audio_dir=data_cfg.get("speaker_audio_dir", "speaker_audios"),
        listener_audio_dir=data_cfg.get("listener_audio_dir", "listener_audios"),
        audio_input_mode=data_cfg.get("audio_input_mode", "mixed"),
        crop_speaker_sec=data_cfg.get("crop_speaker_sec"),
        crop_listener_sec=data_cfg.get("crop_listener_sec"),
        silence_speaker=data_cfg.get("silence_speaker", False),
        silence_listener=data_cfg.get("silence_listener", False),
    )
    train_dataset = Qwen2AudioDataset(
        manifest_path=data_path / data_cfg.get("train_manifest", "splits/train"),
        **common_dataset_kwargs,
    )

    val_dataset = Qwen2AudioDataset(
        manifest_path=data_path / data_cfg.get("val_manifest", "splits/valid"),
        **common_dataset_kwargs,
    )

    logger.info(f"Train: {len(train_dataset)}개 샘플, Valid: {len(val_dataset)}개 샘플")
    logger.info(f"Train 클래스 분포: {train_dataset.get_class_distribution()}")

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
    adapter.load_model()

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

    # === 4. LoRA 설정 ===
    lora_builder = QwenLoraConfigBuilder(
        target_preset=lora_cfg.get("target_preset", "llm_only"),
        rank=lora_cfg.get("rank", 16),
        alpha=lora_cfg.get("alpha", 32),
        dropout=lora_cfg.get("dropout", 0.05),
    )
    lora_config_dict = lora_builder.build()

    # === 5. 학습 설정 ===
    training_config = {
        **train_cfg,
        "use_chat_template": qwen2_cfg.get("use_chat_template", True),
    }
    if max_train_samples is not None:
        training_config["max_train_samples"] = max_train_samples
    if max_val_samples is not None:
        training_config["max_val_samples"] = max_val_samples

    resolved_config = copy.deepcopy(config)
    resolved_config.setdefault("training", {})
    if max_train_samples is not None:
        resolved_config["training"]["max_train_samples"] = max_train_samples
    if max_val_samples is not None:
        resolved_config["training"]["max_val_samples"] = max_val_samples

    fewshot_config, fewshot_selector = build_fewshot_selector(config, train_dataset)

    # === 6. Trainer 구성 + 학습 실행 ===
    trainer = setup_training(
        adapter=adapter,
        dataset=train_dataset,
        val_dataset=val_dataset,
        prompt_builder=prompt_builder,
        lora_config_dict=lora_config_dict,
        training_config=training_config,
        fewshot_selector=fewshot_selector,
        train_variants_per_target=fewshot_config.train_variants_per_target,
        eval_variants_per_target=fewshot_config.eval_variants_per_target,
    )

    # === 6b. (선택) Aim 트래킹 콜백 — opt-in, 기본 비활성 (M5) ===
    # config에 tracking.aim.enabled=true 가 없거나 aim 미설치면 None → 학습 무영향.
    git_info = get_git_info(Path(__file__).resolve().parent.parent)
    aim_hparams = {
        "config_path": config_path,
        "model_id": adapter.model_id,
        "prompt_version": qwen2_cfg.get("prompt_version", "v1"),
        "audio_input_mode": data_cfg.get("audio_input_mode", "mixed"),
        "data_path": str(data_path),
        "lora": lora_cfg,
        "training": training_config,
    }
    aim_callback = build_aim_callback(
        config,
        hparams=aim_hparams,
        git_commit=git_info.get("git_commit"),
    )
    if aim_callback is not None:
        trainer.add_callback(aim_callback)
        logger.info("Aim 트래킹 콜백 등록됨")

    logger.info("=" * 60)
    logger.info("LoRA 학습 시작")
    logger.info(f"  모델: {adapter.model_id}")
    logger.info(f"  LoRA preset: {lora_cfg.get('target_preset', 'llm_only')}")
    logger.info(f"  Rank: {lora_cfg.get('rank', 16)}, Alpha: {lora_cfg.get('alpha', 32)}")
    logger.info(f"  Epochs: {train_cfg.get('num_epochs', 3)}")
    logger.info(f"  Batch size: {train_cfg.get('batch_size', 2)}")
    logger.info(f"  Gradient accumulation: {train_cfg.get('gradient_accumulation_steps', 8)}")
    logger.info(f"  Learning rate: {train_cfg.get('learning_rate', 2e-4)}")
    logger.info("=" * 60)

    start_time = time.time()
    if resume_from_checkpoint:
        logger.info(f"체크포인트에서 재개: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()
    train_time = time.time() - start_time

    logger.info(f"학습 완료: {train_time:.1f}초")

    # === 7. LoRA 가중치 저장 ===
    # DDP / PeftModel 조합에서 load_best_model_at_end 후에도 adapter.model에 best가
    # 반영되지 않는 케이스가 있어 (e.g., SWBD st_w10/w20: trainer state는 epoch1을 best로
    # 기록하지만 save_pretrained가 epoch3 weights를 저장), best checkpoint 디렉토리를
    # 그대로 final/로 직접 복사한다. rank 0만 수행.
    output_dir = Path(train_cfg.get("output_dir", "checkpoints/qwen2_audio_lora"))
    final_dir = output_dir / "final"

    is_main = int(os.environ.get("LOCAL_RANK", "0")) == 0
    best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
    if is_main:
        if best_ckpt and Path(best_ckpt).is_dir():
            logger.info(
                f"best checkpoint를 final로 복사: {best_ckpt} → {final_dir} "
                f"(best_metric={getattr(trainer.state, 'best_metric', None)})"
            )
            if final_dir.exists():
                shutil.rmtree(final_dir)
            shutil.copytree(best_ckpt, final_dir)
            # tokenizer/processor 파일 추가 보장 (best ckpt에는 보통 없음)
            adapter.processor.save_pretrained(str(final_dir))
        else:
            logger.warning(
                "best_model_checkpoint 없음 — 현재 adapter 상태를 final/에 저장 (fallback). "
                "load_best_model_at_end 가 작동하지 않았을 수 있음."
            )
            save_lora_weights(adapter, final_dir)
        save_resolved_config(resolved_config, final_dir / "config.yaml")

    # 학습 메타데이터 저장
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "train_time_sec": round(train_time, 2),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "config_path": config_path,
        **get_git_info(Path(__file__).resolve().parent.parent),
        **adapter.get_metadata(),
        **prompt_builder.get_metadata(),
        **lora_builder.get_metadata(),
        "qwen2_audio_config": qwen2_cfg,
        "data_config": data_cfg,
        "training_config": training_config,
        "inference_config": config.get("inference", {}),
        "dynamic_fewshot": fewshot_config.to_dict(),
    }

    if is_main:
        metadata_path = final_dir / "training_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"메타데이터 저장: {metadata_path}")

    # === 8. 학습 후 평가 (선택) ===
    if train_cfg.get("eval_after_training", True):
        # post-training eval은 generation 기반. DDP 환경에서 adapter.model이 best로 동기화
        # 안 되어 있을 수 있으니, best가 final/에 잘 들어가 있는지 확인 후 adapter에 강제 reload.
        if is_main and final_dir.exists() and best_ckpt:
            try:
                from peft import PeftModel
                base_model = adapter.model.get_base_model() if hasattr(adapter.model, "get_base_model") else adapter.model
                # active adapter weights를 final 경로에서 다시 read
                if hasattr(adapter.model, "load_adapter"):
                    active = "default"
                    if hasattr(adapter.model, "active_adapters"):
                        active = adapter.model.active_adapters[0] if adapter.model.active_adapters else "default"
                    elif hasattr(adapter.model, "active_adapter"):
                        active = adapter.model.active_adapter
                    adapter.model.load_adapter(str(final_dir), active)
                    logger.info(f"post-eval 전 adapter에 best LoRA reload: {final_dir} (adapter={active})")
            except Exception as e:
                logger.warning(f"post-eval 전 best LoRA reload 실패: {e} — 현재 weights로 진행")
        logger.info("학습 후 평가 시작...")
        _run_post_training_eval(
            adapter,
            val_dataset,
            prompt_builder,
            qwen2_cfg,
            output_dir,
            fewshot_selector=fewshot_selector,
            fewshot_config=fewshot_config,
            data_cfg=data_cfg,
        )

    return trainer


def _run_post_training_eval(
    adapter,
    val_dataset,
    prompt_builder,
    qwen2_cfg,
    output_dir,
    fewshot_selector=None,
    fewshot_config=None,
    data_cfg=None,
):
    """학습 완료 후 validation set으로 평가를 수행한다."""
    data_cfg = data_cfg or {}
    valid_labels = data_cfg.get("valid_labels")
    parser_cfg = qwen2_cfg.get("label_parser", {})
    label_parser = QwenLabelParser(
        fallback_label=parser_cfg.get("fallback_label"),
        valid_labels=valid_labels,
    )
    eval_class_names = sorted(valid_labels) if valid_labels else None
    evaluator = Qwen2AudioEvaluator(exclude_invalid=True, class_names=eval_class_names)
    use_chat_template = qwen2_cfg.get("use_chat_template", True)
    audio_input_mode = getattr(val_dataset, "audio_input_mode", "mixed")

    # 모델을 eval 모드로
    adapter.model.eval()

    results = []
    batch_size = 1 if audio_input_mode == "separated" else 4

    for batch_start in range(0, len(val_dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(val_dataset))

        audios = []
        prompts = []
        batch_samples = []
        batch_support_examples = []

        for idx in range(batch_start, batch_end):
            sample = val_dataset[idx]
            batch_samples.append(sample)

            audio_paths = val_dataset.get_audio_paths(sample)
            support_examples = []
            if fewshot_selector is not None:
                support_examples = fewshot_selector.select(
                    target_sample=sample,
                    target_index=idx,
                    variant_id=0,
                )
            batch_support_examples.append(support_examples)

            prompt = prompt_builder.build(
                audio_paths=audio_paths,
                speaker_text=sample.get("speaker_text"),
                support_examples=support_examples,
            )
            prompt_audio_paths = extract_audio_paths_from_prompt(prompt, audio_paths)
            for audio_path in prompt_audio_paths:
                audios.append(val_dataset.load_audio(audio_path))
            prompts.append(prompt)

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

    # 평가 + 저장
    evaluation = evaluator.evaluate(results)
    if fewshot_config is not None:
        evaluation["dynamic_fewshot"] = fewshot_config.to_dict()
    eval_dir = output_dir / "post_training_eval"
    evaluator.save_evaluation(evaluation, eval_dir, split="valid")
    evaluator.print_summary(evaluation, split="valid (post-training)")


def main():
    parser = argparse.ArgumentParser(description="Qwen2-Audio LoRA 학습")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen2_audio.yaml",
        help="설정 파일",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="최대 학습 샘플 수 (디버깅용)",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="최대 검증 샘플 수 (디버깅용)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="HF Trainer 체크포인트 경로에서 학습 재개",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_training(
        config,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        config_path=args.config,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()

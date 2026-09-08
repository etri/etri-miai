# Qwen2-Audio Backchannel Detection

[[Korean](./README.md)] [[English](./README.en.md)]

Qwen2-Audio(7B) 기반 백채널(backchannel) 검출을 위한 학습·추론 코드입니다.
Qwen2-Audio-7B-Instruct 에 LoRA 어댑터를 붙여 대화 중 청자(listener)의 반응 유형을 생성형(generative)으로 분류합니다.

- **입력**: 화자·청자 오디오 (믹싱 또는 분리) + (선택) 화자 발화 텍스트
- **출력**: `NOBC / CONTINUER / UNDERSTANDING / EMPATHY` 중 하나의 클래스 라벨
- **기법**: HuggingFace `Trainer` + PEFT LoRA (audio encoder / LLM / both 선택)

Switchboard(SWBD) 데이터셋을 위한 3-class(`NOBC / CONTINUER / ASSESSMENT`)
설정도 포함되어 있습니다.

## 프로젝트 구조

```
bc_detection_public/
├── src/bc_detection/
│   ├── qwen2/                    # 핵심 모듈 (adapter, dataset, trainer, prompt_builder, ...)
│   │   ├── adapter.py            # Qwen2-Audio 모델/프로세서 로딩
│   │   ├── constants.py          # 라벨/샘플레이트/LoRA target preset
│   │   ├── dataset.py            # 매니페스트 기반 데이터셋
│   │   ├── prompt_builder.py     # chat template + fewshot 프롬프트 구성
│   │   ├── label_parser.py       # 모델 출력 → 정규화 라벨
│   │   ├── lora_config.py        # PEFT LoRA config 빌더
│   │   ├── trainer.py            # HF Trainer / DataCollator
│   │   ├── evaluator.py          # metric + confusion matrix
│   │   ├── fewshot.py            # dynamic few-shot selector
│   │   ├── aim_logging.py        # (opt) Aim tracking callback
│   │   ├── metadata.py           # git info / resolved config 저장
│   │   └── mlflow_pyfunc.py      # (opt) MLflow PyFunc 래퍼
│   └── utils/                    # 샘플링 · 메트릭 유틸
├── scripts/
│   ├── train_qwen2_audio.py      # LoRA fine-tuning 스크립트
│   └── inference_qwen2_audio.py  # 추론 스크립트 (단일/torchrun 다중 GPU)
├── configs/
│   ├── qwen2_audio_v1.yaml               # BCKO 4-class 기본 설정 (mixed audio)
│   ├── qwen2_audio_v1_audio_only.yaml    # audio-only 통제 실험
│   ├── qwen2_audio_v1_lora_rank8.yaml    # LoRA rank 민감도 (rank=8)
│   └── qwen2_audio_v1_swbd_baseline.yaml # SWBD 3-class 영어 baseline
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## 요구사항

- Python 3.10+
- CUDA 지원 GPU (Qwen2-Audio-7B 로딩에 최소 ~16GB VRAM, LoRA 학습 권장 24GB+)
- PyTorch 2.x + `transformers>=4.30`, `accelerate`, `peft`

## 설치

```bash
# 1) 가상환경 (예시)
conda create -n bc_qwen2 python=3.10 -y
conda activate bc_qwen2

# 2) 설치 (editable)
pip install -e .

# (선택) 실험 트래킹 / MLflow 사용 시
pip install -e ".[tracking]"   # Aim 3.24.0
pip install -e ".[mlflow]"     # MLflow >= 2.9
```

## 라벨 스키마

기본(BCKO 4-class):

| 원본 라벨 | 최종 클래스 |
|-----------|-------------|
| ALONE / BARGIN | NOBC |
| CONTINUER | CONTINUER |
| UNDERSTANDING | UNDERSTANDING |
| ASSESSMENT / AFFIRMATIVE / NEGATIVE_SURPRISE / POSITIVE_SURPRISE / REQUEST_CONFIRMATION | EMPATHY |

SWBD 3-class:

| 원본 라벨 | 최종 클래스 |
|-----------|-------------|
| ALONE / BARGIN | NOBC |
| CONTINUER | CONTINUER |
| ASSESSMENT | ASSESSMENT |

라벨 매핑과 `valid_labels` 는 config 의 `data.label_mapping` / `data.valid_labels` 로 오버라이드할 수 있습니다.

## 데이터 형식

`Qwen2AudioDataset` 은 매니페스트 3-tuple 을 읽습니다:

```
<data_path>/
├── splits/
│   ├── train.label            # "<sample_id> <RAW_LABEL>" 한 줄씩
│   ├── train.speaker.text     # "<sample_id> <speaker text>" 한 줄씩
│   ├── train.listener.text    # "<sample_id> <listener text>" 한 줄씩
│   ├── valid.label
│   ├── valid.speaker.text
│   ├── valid.listener.text
│   ├── test.label
│   ├── test.speaker.text
│   └── test.listener.text
├── mixing_audios/             # <sample_id>.wav (16kHz, mono)  # audio_input_mode="mixed"
├── speaker_audios/            # <sample_id>.wav (16kHz, mono)  # audio_input_mode="separated"
└── listener_audios/           # <sample_id>.wav (16kHz, mono)  # audio_input_mode="separated"
```

- `audio_input_mode: "mixed"` 는 `mixing_audios/` 하나만 사용합니다.
- `audio_input_mode: "separated"` 는 speaker/listener 오디오를 각각 audio token 으로 넣습니다.

## 학습

```bash
# 단일 GPU
python scripts/train_qwen2_audio.py --config configs/qwen2_audio_v1.yaml

# 다중 GPU (torchrun, DDP)
torchrun --standalone --nproc_per_node=4 \
    scripts/train_qwen2_audio.py --config configs/qwen2_audio_v1.yaml

# 빠른 스모크 (train/val 각각 소수만)
python scripts/train_qwen2_audio.py \
    --config configs/qwen2_audio_v1.yaml \
    --max_train_samples 100 --max_val_samples 50
```

주요 config 스위치:

- `qwen2_audio.model_type`: `"instruct"` (권장) 또는 `"base"`
- `qwen2_audio.lora.target_preset`: `"audio_only" / "llm_only" / "both"`
- `qwen2_audio.lora.rank / alpha / dropout`
- `qwen2_audio.use_speaker_text`, `use_listener_text`: 텍스트 프롬프트 포함 여부
- `qwen2_audio.prompt_version` + `qwen2_audio.prompt_templates.<key>`: prompt 스왑
- `qwen2_audio.fewshot` (정적 K-shot) 또는 `qwen2_audio.dynamic_fewshot` (동적 selector)
- `data.audio_input_mode`: `"mixed" / "separated"`
- `training.balanced_sampling`: `WeightedRandomSampler` 활성화
- `training.gradient_checkpointing`: VRAM 절약
- `training.report_to`: `"none" / "tensorboard" / "wandb"`
- `tracking.aim.{enabled,repo,experiment}`: (opt) Aim 서버 트래킹

## 추론

```bash
# 단일 GPU
python scripts/inference_qwen2_audio.py \
    --config configs/qwen2_audio_v1.yaml \
    --split test

# 다중 GPU (samples[rank::world_size] shard 후 rank-0 이 merge & evaluate)
torchrun --standalone --nproc_per_node=4 \
    scripts/inference_qwen2_audio.py \
    --config configs/qwen2_audio_v1.yaml --split test

# 최대 샘플 제한 (디버깅)
python scripts/inference_qwen2_audio.py \
    --config configs/qwen2_audio_v1.yaml --split test --max_samples 32
```

추론 결과는 `inference.output_dir` 아래에 JSONL(또는 CSV)로 저장되고,
metric(accuracy / macro-F1 / per-class F1 / confusion matrix)이 함께 기록됩니다.

학습된 LoRA 가중치를 로드하려면 config 의
`qwen2_audio.lora.weights_path` 에 checkpoint 디렉토리 경로를 지정하세요.

## 모델 라이선스

- 코드: 이 저장소 자체는 MIT License.
- 백본 모델(`Qwen/Qwen2-Audio-7B` / `Qwen/Qwen2-Audio-7B-Instruct`)은
  Alibaba 의 별도 라이선스를 따릅니다. HuggingFace 페이지의 라이선스 조항을 확인하세요.

## 인용

논문/기술문서에서 이 코드를 인용하실 경우, 이 저장소 URL 과 함께
Qwen2-Audio 원 저자(Alibaba DAMO Academy)의 논문을 함께 인용해 주세요.

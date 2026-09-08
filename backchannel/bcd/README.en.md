# Qwen2-Audio Backchannel Detection

[[Korean](./README.md)] [[English](./README.en.md)]

Training and inference code for backchannel detection based on Qwen2-Audio (7B).
A LoRA adapter is attached to Qwen2-Audio-7B-Instruct to classify a listener's
reaction type during conversation in a generative manner.

- **Input**: speaker/listener audio (mixed or separated) + (optional) speaker utterance text
- **Output**: one class label among `NOBC / CONTINUER / UNDERSTANDING / EMPATHY`
- **Method**: HuggingFace `Trainer` + PEFT LoRA (choose audio encoder / LLM / both)

A 3-class (`NOBC / CONTINUER / ASSESSMENT`) setup for the Switchboard (SWBD)
dataset is also included.

## Project Structure

```
bc_detection_public/
├── src/bc_detection/
│   ├── qwen2/                    # core modules (adapter, dataset, trainer, prompt_builder, ...)
│   │   ├── adapter.py            # Qwen2-Audio model/processor loading
│   │   ├── constants.py          # labels / sample rate / LoRA target preset
│   │   ├── dataset.py            # manifest-based dataset
│   │   ├── prompt_builder.py     # chat template + fewshot prompt construction
│   │   ├── label_parser.py       # model output → normalized label
│   │   ├── lora_config.py        # PEFT LoRA config builder
│   │   ├── trainer.py            # HF Trainer / DataCollator
│   │   ├── evaluator.py          # metric + confusion matrix
│   │   ├── fewshot.py            # dynamic few-shot selector
│   │   ├── aim_logging.py        # (opt) Aim tracking callback
│   │   ├── metadata.py           # git info / resolved config saving
│   │   └── mlflow_pyfunc.py      # (opt) MLflow PyFunc wrapper
│   └── utils/                    # sampling · metric utilities
├── scripts/
│   ├── train_qwen2_audio.py      # LoRA fine-tuning script
│   └── inference_qwen2_audio.py  # inference script (single / torchrun multi-GPU)
├── configs/
│   ├── qwen2_audio_v1.yaml               # BCKO 4-class default config (mixed audio)
│   ├── qwen2_audio_v1_audio_only.yaml    # audio-only control experiment
│   ├── qwen2_audio_v1_lora_rank8.yaml    # LoRA rank sensitivity (rank=8)
│   └── qwen2_audio_v1_swbd_baseline.yaml # SWBD 3-class English baseline
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (at least ~16GB VRAM to load Qwen2-Audio-7B; 24GB+ recommended for LoRA training)
- PyTorch 2.x + `transformers>=4.30`, `accelerate`, `peft`

## Installation

```bash
# 1) virtual environment (example)
conda create -n bc_qwen2 python=3.10 -y
conda activate bc_qwen2

# 2) install (editable)
pip install -e .

# (optional) experiment tracking / MLflow
pip install -e ".[tracking]"   # Aim 3.24.0
pip install -e ".[mlflow]"     # MLflow >= 2.9
```

## Label Schema

Default (BCKO 4-class):

| Original Label | Final Class |
|----------------|-------------|
| ALONE / BARGIN | NOBC |
| CONTINUER | CONTINUER |
| UNDERSTANDING | UNDERSTANDING |
| ASSESSMENT / AFFIRMATIVE / NEGATIVE_SURPRISE / POSITIVE_SURPRISE / REQUEST_CONFIRMATION | EMPATHY |

SWBD 3-class:

| Original Label | Final Class |
|----------------|-------------|
| ALONE / BARGIN | NOBC |
| CONTINUER | CONTINUER |
| ASSESSMENT | ASSESSMENT |

Label mapping and `valid_labels` can be overridden via `data.label_mapping` /
`data.valid_labels` in the config.

## Data Format

`Qwen2AudioDataset` reads a 3-tuple manifest:

```
<data_path>/
├── splits/
│   ├── train.label            # one "<sample_id> <RAW_LABEL>" per line
│   ├── train.speaker.text     # one "<sample_id> <speaker text>" per line
│   ├── train.listener.text    # one "<sample_id> <listener text>" per line
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

- `audio_input_mode: "mixed"` uses only `mixing_audios/`.
- `audio_input_mode: "separated"` feeds speaker/listener audios as separate audio tokens.

## Training

```bash
# single GPU
python scripts/train_qwen2_audio.py --config configs/qwen2_audio_v1.yaml

# multi-GPU (torchrun, DDP)
torchrun --standalone --nproc_per_node=4 \
    scripts/train_qwen2_audio.py --config configs/qwen2_audio_v1.yaml

# quick smoke test (small subsets of train/val)
python scripts/train_qwen2_audio.py \
    --config configs/qwen2_audio_v1.yaml \
    --max_train_samples 100 --max_val_samples 50
```

Key config switches:

- `qwen2_audio.model_type`: `"instruct"` (recommended) or `"base"`
- `qwen2_audio.lora.target_preset`: `"audio_only" / "llm_only" / "both"`
- `qwen2_audio.lora.rank / alpha / dropout`
- `qwen2_audio.use_speaker_text`, `use_listener_text`: whether to include text prompts
- `qwen2_audio.prompt_version` + `qwen2_audio.prompt_templates.<key>`: prompt swap
- `qwen2_audio.fewshot` (static K-shot) or `qwen2_audio.dynamic_fewshot` (dynamic selector)
- `data.audio_input_mode`: `"mixed" / "separated"`
- `training.balanced_sampling`: enable `WeightedRandomSampler`
- `training.gradient_checkpointing`: save VRAM
- `training.report_to`: `"none" / "tensorboard" / "wandb"`
- `tracking.aim.{enabled,repo,experiment}`: (opt) Aim server tracking

## Inference

```bash
# single GPU
python scripts/inference_qwen2_audio.py \
    --config configs/qwen2_audio_v1.yaml \
    --split test

# multi-GPU (samples[rank::world_size] sharding; rank-0 merges & evaluates)
torchrun --standalone --nproc_per_node=4 \
    scripts/inference_qwen2_audio.py \
    --config configs/qwen2_audio_v1.yaml --split test

# limit max samples (debugging)
python scripts/inference_qwen2_audio.py \
    --config configs/qwen2_audio_v1.yaml --split test --max_samples 32
```

Inference results are saved under `inference.output_dir` as JSONL (or CSV), and
metrics (accuracy / macro-F1 / per-class F1 / confusion matrix) are recorded together.

To load trained LoRA weights, set the checkpoint directory path in
`qwen2_audio.lora.weights_path` of the config.

## Model License

- Code: this repository itself is under the MIT License.
- Backbone models (`Qwen/Qwen2-Audio-7B` / `Qwen/Qwen2-Audio-7B-Instruct`) follow
  Alibaba's separate license. Please check the license terms on the HuggingFace page.

## Citation

When citing this code in papers/technical documents, please cite the original
Qwen2-Audio paper by the authors (Alibaba DAMO Academy) along with this repository URL.

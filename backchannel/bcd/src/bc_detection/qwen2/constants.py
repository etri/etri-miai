"""
Qwen2-Audio 백채널 검출 공통 상수
"""

# 백채널 클래스
VALID_LABELS = ["NOBC", "CONTINUER", "UNDERSTANDING", "EMPATHY"]
LABEL_CHOICES_STR = " / ".join(VALID_LABELS)
CLASS_NAMES = sorted(VALID_LABELS)
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASS_NAMES)}

# 오디오 상수 (기존 preprocessing.py와 동일)
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION = 1.5

# 학습
IGNORE_INDEX = -100  # CrossEntropyLoss에서 무시할 인덱스

# 모델 ID
MODEL_REGISTRY = {
    "base": "Qwen/Qwen2-Audio-7B",
    "instruct": "Qwen/Qwen2-Audio-7B-Instruct",
}

# LoRA target module (PEFT regex 패턴)
# 실제 모듈 경로:
#   Audio Encoder: audio_tower.layers.{N}.self_attn.{q,v}_proj
#   LLM:           language_model.model.layers.{N}.self_attn.{q,v}_proj
AUDIO_ENCODER_TARGETS = r"audio_tower\.layers\.\d+\.self_attn\.[qv]_proj"
LLM_TARGETS = r"language_model\.model\.layers\.\d+\.self_attn\.[qv]_proj"
BOTH_TARGETS = r"(audio_tower\.layers\.\d+|language_model\.model\.layers\.\d+)\.self_attn\.[qv]_proj"

LORA_TARGET_PRESETS = {
    "audio_only": AUDIO_ENCODER_TARGETS,
    "llm_only": LLM_TARGETS,
    "both": BOTH_TARGETS,
}

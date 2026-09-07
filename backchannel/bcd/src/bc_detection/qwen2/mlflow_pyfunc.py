"""MLflow custom pyfunc wrapper for Qwen2-Audio LoRA adapters (M10).

원칙 (docs/plan §14):
- base weight(`Qwen/Qwen2-Audio-7B-Instruct`, ~16GB)는 artifact에 넣지 않는다.
  추론 노드의 HF 캐시(또는 네트워크)에서 로드한다.
- artifact에는 LoRA adapter(`adapter_model.safetensors` + `adapter_config.json` +
  tokenizer)와 학습 시점의 `config.yaml`만 담는다.
- 이 pyfunc는 "predictor"가 아니라 "loader"다. `load_context`가 base+adapter를
  결합한 `Qwen2AudioAdapter`를 만들어 `self.adapter`로 노출한다. 클라이언트는
  `mlflow.pyfunc.load_model(...).unwrap_python_model()`로 이 인스턴스를 복구해
  기존 `scripts/inference_qwen2_audio.py`의 generation 경로를 그대로 재사용한다.
  (오디오 배열/processor/prompt 빌드는 pyfunc `predict`의 DataFrame 계약에
  맞지 않으므로 predict 안에 전체 파이프라인을 재구현하지 않는다.)

mlflow 미설치 환경(학습 등)에서도 import 가능하도록 base 클래스는 soft-import한다.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:  # mlflow는 추론/등록 노드에만 있으면 된다.
    from mlflow.pyfunc import PythonModel as _PythonModel
except Exception:  # pragma: no cover - mlflow 미설치 환경
    _PythonModel = object  # type: ignore[assignment, misc]


# artifact 키 (register 스크립트와 공유)
ARTIFACT_ADAPTER_DIR = "adapter_dir"
ARTIFACT_CONFIG_YAML = "config_yaml"


class Qwen2AudioLoRAModel(_PythonModel):
    """Qwen2-Audio base(HF 캐시) + LoRA adapter(artifact)를 결합해 로드하는 pyfunc.

    `load_context` 후 `self.adapter`(로드된 `Qwen2AudioAdapter`)와
    `self.config`(학습 config dict)가 노출된다.
    """

    def load_context(self, context):
        import yaml

        from bc_detection.qwen2.adapter import Qwen2AudioAdapter

        adapter_dir = context.artifacts[ARTIFACT_ADAPTER_DIR]
        config_yaml = context.artifacts[ARTIFACT_CONFIG_YAML]

        with open(config_yaml, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        qwen2_cfg = self.config["qwen2_audio"]

        # inference_qwen2_audio.py:212 와 동일한 인스턴스화 계약을 유지한다.
        self.adapter = Qwen2AudioAdapter(
            model_type=qwen2_cfg.get("model_type", "instruct"),
            model_id=qwen2_cfg.get("model_id"),
            device_map=qwen2_cfg.get("device_map", "auto"),
            torch_dtype=qwen2_cfg.get("torch_dtype", "float16"),
            max_new_tokens=qwen2_cfg.get("max_new_tokens", 32),
            do_sample=qwen2_cfg.get("do_sample", False),
            attn_implementation=qwen2_cfg.get("attn_implementation", "sdpa"),
        )
        self.adapter.load_model()  # base = HF 캐시
        self.adapter.load_lora_weights(adapter_dir)  # adapter = artifact
        logger.info("Qwen2AudioLoRAModel load_context 완료: adapter_dir=%s", adapter_dir)

    def predict(self, context, model_input, params=None):
        """이 모델은 loader다. 실추론은 unwrap_python_model()로 adapter를 복구해 수행한다.

        pyfunc `predict` 계약(DataFrame/ndarray)은 오디오+processor+prompt 빌드에
        맞지 않으므로 여기서 전체 파이프라인을 재구현하지 않는다.
        """
        raise NotImplementedError(
            "Qwen2AudioLoRAModel은 loader 전용입니다. "
            "`mlflow.pyfunc.load_model(uri).unwrap_python_model().adapter`로 "
            "Qwen2AudioAdapter를 복구해 기존 generation 경로를 사용하세요."
        )


def build_artifacts(experiment_dir: str | Path) -> dict:
    """experiment_dir(final/ 포함)에서 pyfunc artifacts dict를 구성한다.

    base weight는 포함하지 않는다. adapter_dir 안에는 adapter_model.safetensors,
    adapter_config.json, tokenizer 파일이 있어야 한다.
    """
    exp = Path(experiment_dir)
    final_dir = exp / "final"
    config_yaml = final_dir / "config.yaml"
    adapter_file = final_dir / "adapter_model.safetensors"
    adapter_config = final_dir / "adapter_config.json"

    missing = [str(p) for p in (config_yaml, adapter_file, adapter_config) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"필수 artifact 누락: {missing}")

    return {
        ARTIFACT_ADAPTER_DIR: str(final_dir),
        ARTIFACT_CONFIG_YAML: str(config_yaml),
    }

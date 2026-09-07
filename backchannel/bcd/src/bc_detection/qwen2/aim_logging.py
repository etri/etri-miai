"""Optional Aim experiment tracking for Qwen2-Audio training (M5, opt-in).

기본 비활성. config로 켠다:

    tracking:
      aim:
        enabled: true
        repo: "aim://192.168.5.254:53800"   # 생략 시 이 기본값
        experiment: "qwen2_bc"              # 생략 가능

범위 (M5 = 최소 통합):
  - run 생성 + hparams + loss/eval_loss curve + git_commit 태그 + 정상 close.
  - §12의 전체 태그 스키마(corpus/phase/dataset_version/migration_status ...)는
    M8(backfill)로 미룬다.

안전성:
  - 비활성이거나 `aim` 미설치면 build_aim_callback()은 None을 반환하고 학습은
    전혀 영향받지 않는다(추가 import 비용/동작 변화 없음). → flag off = 즉시 revert.
  - main DDP 프로세스(state.is_world_process_zero)에서만 Run을 만들어, 멀티 GPU
    학습에서도 run이 정확히 1개 기록된다.
  - transformers는 soft-import: aim만 있는 환경(스모크 컨테이너)에서도 이 모듈이
    로드된다. transformers가 있으면 정식 TrainerCallback 서브클래스로 동작한다.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_AIM_REPO = "aim://192.168.5.254:53800"
DEFAULT_EXPERIMENT = "qwen2_bc"


def route_metric(key: str):
    """HF Trainer log key → (metric_name, aim context). backfill(M8)과 공유해
    과거/실시간 curve가 UI에서 겹쳐 보이도록 라우팅 규칙을 단일화한다.

    eval_* → context {subset: eval} (접두사 제거), 그 외 → {subset: train}.
    """
    if key.startswith("eval_"):
        return key[len("eval_"):], {"subset": "eval"}
    return key, {"subset": "train"}

try:  # soft dep — 스모크는 transformers 없이 aim만으로 콜백 경로를 구동한다
    from transformers import TrainerCallback as _TrainerCallbackBase
except Exception:  # pragma: no cover - transformers 미설치 환경
    _TrainerCallbackBase = object


class AimTrainerCallback(_TrainerCallbackBase):
    """HF Trainer metric을 Aim run으로 기록한다 (rank-0 전용)."""

    def __init__(
        self,
        repo: str,
        experiment: str,
        hparams: Optional[dict] = None,
        git_commit: Optional[str] = None,
    ):
        self._repo = repo
        self._experiment = experiment
        self._hparams = hparams or {}
        self._git_commit = git_commit
        self._run = None

    # HF Trainer가 이미 rank를 알고 있으므로 env-var로 추측하지 않고 state를 쓴다.
    def on_train_begin(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return  # 비-main rank는 Run 자체를 만들지 않음
        from aim import Run

        # DDP-safety: aim의 백그라운드 스레드(터미널 로그 캡처 / 시스템 리소스 트래커)를
        # 끈다. 이들이 살아 있으면 멀티프로세스(torchrun) 학습이 teardown에서 hang한다
        # (실측: 콜백 켜면 2-GPU DDP가 teardown에서 멈춤, 끄면 정상 종료).
        self._run = Run(
            repo=self._repo,
            experiment=self._experiment,
            capture_terminal_logs=False,
            system_tracking_interval=None,
            log_system_params=False,
        )
        if self._hparams:
            self._run["hparams"] = self._hparams
        if self._git_commit:
            self._run["git_commit"] = self._git_commit
        logger.info(
            "[aim] run 시작 hash=%s repo=%s experiment=%s",
            self._run.hash,
            self._repo,
            self._experiment,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._run is None or not logs:
            return
        step = int(getattr(state, "global_step", 0) or 0)
        for key, value in logs.items():
            # bool은 int의 서브클래스라 명시적으로 제외
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            name, context = route_metric(key)
            self._run.track(float(value), name=name, step=step, context=context)

    def on_train_end(self, args, state, control, **kwargs):
        if self._run is not None:
            self._run.close()
            logger.info("[aim] run 종료")
            self._run = None


def build_aim_callback(
    config,
    hparams: Optional[dict] = None,
    git_commit: Optional[str] = None,
    experiment: Optional[str] = None,
):
    """tracking.aim.enabled면 Aim TrainerCallback을, 아니면 None을 반환한다.

    tracking 블록이 없거나 비활성이거나 `aim` 미설치면 None(no-op)을 반환하며,
    이 경우 학습 동작은 변하지 않는다.
    """
    tracking = (config or {}).get("tracking") or {}
    aim_cfg = tracking.get("aim") or {}
    if not aim_cfg.get("enabled", False):
        return None

    try:
        import aim  # noqa: F401
    except Exception as exc:  # aim 미설치 → 조용히 no-op (경고만)
        logger.warning(
            "[aim] tracking.aim.enabled=true 이나 aim 미설치(%s) → 트래킹 건너뜀", exc
        )
        return None

    repo = aim_cfg.get("repo") or DEFAULT_AIM_REPO
    experiment = experiment or aim_cfg.get("experiment") or DEFAULT_EXPERIMENT
    logger.info("[aim] tracking 활성화 repo=%s experiment=%s", repo, experiment)
    return AimTrainerCallback(
        repo=repo,
        experiment=experiment,
        hparams=hparams,
        git_commit=git_commit,
    )

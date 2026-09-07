"""Metadata helpers for Qwen2-Audio experiment reproducibility."""

import logging
import subprocess
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


def get_git_info(repo_root: Optional[Path] = None) -> dict:
    """Return git branch/commit info without failing outside a git repo."""
    root = Path(repo_root or Path.cwd())

    def run_git(args):
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=str(root),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return None

    return {
        "git_commit": run_git(["rev-parse", "HEAD"]),
        "git_branch": run_git(["branch", "--show-current"]),
        "git_dirty": bool(run_git(["status", "--short"])),
    }


def save_resolved_config(config: dict, output_path: Path) -> None:
    """Save the resolved YAML config used for an experiment."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
    logger.info("Resolved config 저장: %s", output_path)

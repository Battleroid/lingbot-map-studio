from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from app.config import settings
from app.jobs.schema import JobEvent

log = logging.getLogger(__name__)

# Per-processor checkpoint registry.
#
# Each entry maps (processor_id, model_id) → (repo_id, filename). The cache
# lives at `<models_dir>/checkpoints/<processor_id>/<filename>` so backends
# with conflicting filenames coexist cleanly and `worker-slam` doesn't see
# lingbot weights cluttering its cache dir.
#
# SLAM entries are placeholders: the simulated Phase-4 tracker doesn't
# actually need weights, but when the real upstream integrations land the
# filenames/repos here are the single place to update.
_REGISTRY: dict[tuple[str, str], tuple[str, str]] = {
    ("lingbot", "lingbot-map"): (settings.hf_repo_id, "lingbot-map.pt"),
    ("lingbot", "lingbot-map-long"): (settings.hf_repo_id, "lingbot-map-long.pt"),
    ("lingbot", "lingbot-map-stage1"): (settings.hf_repo_id, "lingbot-map-stage1.pt"),
    # SLAM backend defaults — no-op until real integrations land.
    ("droid_slam", "default"): ("anthropic/lingbot-map-studio-models", "droid_slam.pth"),
    ("mast3r_slam", "default"): ("anthropic/lingbot-map-studio-models", "mast3r_slam.pth"),
    ("dpvo", "default"): ("anthropic/lingbot-map-studio-models", "dpvo.pth"),
    ("monogs", "default"): ("anthropic/lingbot-map-studio-models", "monogs.pth"),
}


async def ensure_checkpoint(
    model_id: str,
    job_id: str,
    publish,
    *,
    processor_id: str = "lingbot",
    optional: bool = False,
) -> Optional[Path]:
    """Lazily download a checkpoint into the shared /models volume.

    Generalised to key by `(processor_id, model_id)` so every backend caches
    under its own subdirectory and name collisions between backends are
    impossible.

    `optional=True` makes this a best-effort fetch — if the repo/filename
    isn't registered or the download fails, return None instead of
    raising. Useful for SLAM backends that can fall back to their
    simulated tracker when weights are missing.
    """
    key = (processor_id, model_id)
    entry = _REGISTRY.get(key)
    if entry is None:
        if optional:
            await publish(
                JobEvent(
                    job_id=job_id,
                    stage="checkpoint",
                    level="warn",
                    message=(
                        f"no checkpoint registered for {processor_id}/{model_id}; "
                        "continuing without weights"
                    ),
                )
            )
            return None
        raise ValueError(f"Unknown checkpoint: {processor_id}/{model_id}")

    repo_id, filename = entry
    target_dir = settings.models_dir / "checkpoints" / processor_id
    target_dir.mkdir(parents=True, exist_ok=True)
    expected = target_dir / filename
    if expected.exists():
        await publish(
            JobEvent(
                job_id=job_id,
                stage="checkpoint",
                message=f"checkpoint cached: {expected}",
                data={"path": str(expected), "cached": True, "processor": processor_id},
            )
        )
        return expected

    await publish(
        JobEvent(
            job_id=job_id,
            stage="checkpoint",
            message=f"downloading {filename} from {repo_id}",
            data={"repo": repo_id, "filename": filename, "processor": processor_id},
        )
    )

    def _download() -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )

    try:
        path_str = await asyncio.to_thread(_download)
    except Exception as exc:  # noqa: BLE001
        if optional:
            await publish(
                JobEvent(
                    job_id=job_id,
                    stage="checkpoint",
                    level="warn",
                    message=f"optional checkpoint fetch failed: {exc}",
                )
            )
            return None
        raise
    path = Path(path_str)
    await publish(
        JobEvent(
            job_id=job_id,
            stage="checkpoint",
            message=f"checkpoint ready: {path}",
            data={"path": str(path), "cached": False, "processor": processor_id},
            progress=1.0,
        )
    )
    return path

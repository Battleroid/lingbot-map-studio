from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.config import settings
from app.jobs.schema import JobEvent

log = logging.getLogger(__name__)

# HF repo layout (verified): flat .pt files keyed by model id.
_FILENAMES = {
    "lingbot-map": "lingbot-map.pt",
    "lingbot-map-long": "lingbot-map-long.pt",
    "lingbot-map-stage1": "lingbot-map-stage1.pt",
}


async def ensure_checkpoint(
    model_id: str,
    job_id: str,
    publish,
) -> Path:
    """Lazily download a lingbot-map checkpoint into the shared /models volume.

    hf_hub_download is blocking; run it in a thread so we keep serving WS
    clients. Progress is not fine-grained (huggingface_hub caches internally),
    but we emit start/done events so the UI can show spinners.
    """
    filename = _FILENAMES.get(model_id)
    if filename is None:
        raise ValueError(f"Unknown model_id: {model_id}")

    target_dir = settings.models_dir / "checkpoints"
    target_dir.mkdir(parents=True, exist_ok=True)
    expected = target_dir / filename
    if expected.exists():
        await publish(
            JobEvent(
                job_id=job_id,
                stage="checkpoint",
                message=f"checkpoint cached: {expected}",
                data={"path": str(expected), "cached": True},
            )
        )
        return expected

    await publish(
        JobEvent(
            job_id=job_id,
            stage="checkpoint",
            message=f"downloading {filename} from {settings.hf_repo_id}",
            data={"repo": settings.hf_repo_id, "filename": filename},
        )
    )

    def _download() -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=settings.hf_repo_id,
            filename=filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )

    path_str = await asyncio.to_thread(_download)
    path = Path(path_str)
    await publish(
        JobEvent(
            job_id=job_id,
            stage="checkpoint",
            message=f"checkpoint ready: {path}",
            data={"path": str(path), "cached": False},
            progress=1.0,
        )
    )
    return path

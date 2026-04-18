from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from app.config import settings

log = logging.getLogger(__name__)

DRAFT_TTL_SECONDS = 24 * 3600  # drafts older than a day are swept on access


def _drafts_root() -> Path:
    root = settings.data_dir / "drafts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def draft_dir(draft_id: str) -> Path:
    return _drafts_root() / draft_id


def draft_uploads(draft_id: str) -> Path:
    return draft_dir(draft_id) / "uploads"


def _probe_path(draft_id: str) -> Path:
    return draft_dir(draft_id) / "probe.json"


def save_draft(
    draft_id: str,
    uploads: list[Path],
    probes: list[dict[str, Any]],
    suggested_config: dict[str, Any],
) -> dict[str, Any]:
    rec = {
        "id": draft_id,
        "created_at": time.time(),
        "uploads": [u.name for u in uploads],
        "probes": probes,
        "suggested_config": suggested_config,
    }
    _probe_path(draft_id).write_text(json.dumps(rec, indent=2))
    return rec


def load_draft(draft_id: str) -> dict[str, Any] | None:
    p = _probe_path(draft_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def draft_video_paths(draft_id: str) -> list[Path]:
    ups = draft_uploads(draft_id)
    if not ups.exists():
        return []
    return sorted([p for p in ups.iterdir() if p.is_file()])


def delete_draft(draft_id: str) -> bool:
    d = draft_dir(draft_id)
    if not d.exists():
        return False
    shutil.rmtree(d, ignore_errors=True)
    return True


def sweep_expired(now: float | None = None) -> int:
    """Remove drafts older than DRAFT_TTL_SECONDS. Returns count removed."""
    now = now or time.time()
    root = _drafts_root()
    removed = 0
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        pp = entry / "probe.json"
        try:
            rec = json.loads(pp.read_text()) if pp.exists() else None
        except json.JSONDecodeError:
            rec = None
        created = rec.get("created_at") if rec else entry.stat().st_mtime
        if now - created > DRAFT_TTL_SECONDS:
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
    return removed

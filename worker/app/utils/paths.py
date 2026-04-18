from __future__ import annotations

import re
import uuid
from pathlib import Path


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str) -> str:
    stem = Path(name).name
    cleaned = _SAFE.sub("_", stem).strip("._-")
    return cleaned or f"file_{uuid.uuid4().hex[:8]}"


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]

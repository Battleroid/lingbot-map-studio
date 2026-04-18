from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_REV_RE = re.compile(r"^rev_(\d+)\.glb$")


def next_revision(artifacts_dir: Path) -> int:
    if not artifacts_dir.exists():
        return 1
    highest = 0
    for p in artifacts_dir.iterdir():
        m = _REV_RE.match(p.name)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


def latest_revision_path(artifacts_dir: Path) -> Optional[Path]:
    """Return the newest rev_N.glb, or fall back to reconstruction.glb."""
    if not artifacts_dir.exists():
        return None
    newest_rev = 0
    newest: Optional[Path] = None
    for p in artifacts_dir.iterdir():
        m = _REV_RE.match(p.name)
        if m:
            n = int(m.group(1))
            if n > newest_rev:
                newest_rev = n
                newest = p
    if newest is not None:
        return newest
    base = artifacts_dir / "reconstruction.glb"
    return base if base.exists() else None


def revision_number(path: Path) -> int:
    m = _REV_RE.match(path.name)
    return int(m.group(1)) if m else 0


def revision_path(artifacts_dir: Path, revision: int) -> Path:
    return artifacts_dir / f"rev_{revision:03d}.glb"

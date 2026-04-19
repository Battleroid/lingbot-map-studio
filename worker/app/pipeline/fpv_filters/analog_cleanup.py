"""Extra ffmpeg filters for analog / VHS-era FPV footage.

The existing `build_ingest_filters` already emits `hqdn3d` + `deflicker`
when `preproc_denoise` is enabled. This module adds:

  * `atadenoise` — adaptive temporal average; great on chroma noise but
    expensive, so gated behind an explicit `preproc_analog_cleanup` flag.
  * a standalone `deflicker` when the user wants brightness stabilisation
    without the full denoise pair.

Returned snippet is spliced into the ffmpeg chain by `ingest.py`. No
output = this stage is a no-op for that config.
"""

from __future__ import annotations

from typing import Any

from app.jobs.schema import PreprocFields


def ffmpeg_snippet(cfg: PreprocFields) -> list[str]:
    """Return zero or more ffmpeg -vf filter clauses from the config."""
    out: list[str] = []
    if cfg.preproc_analog_cleanup:
        # atadenoise defaults: s=0.02, p=all, tuned for ~8-bit analog grain.
        # Slightly stronger chroma threshold because VHS chroma noise is the
        # dominant artefact we're trying to kill.
        out.append("atadenoise=0a=0.02:0b=0.04:1a=0.02:1b=0.04:2a=0.02:2b=0.04")
    if cfg.preproc_deflicker and not cfg.preproc_denoise:
        # `preproc_denoise` already includes deflicker; avoid stacking two.
        out.append("deflicker=mode=pm:size=5")
    return out


def as_summary(cfg: PreprocFields) -> dict[str, Any]:
    """Small payload for the UI's preproc preview: what will actually fire."""
    return {
        "analog_cleanup": bool(cfg.preproc_analog_cleanup),
        "deflicker": bool(cfg.preproc_deflicker or cfg.preproc_denoise),
        "denoise": bool(cfg.preproc_denoise),
        "filters": ffmpeg_snippet(cfg),
    }

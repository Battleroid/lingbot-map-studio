"""Strict-no-fallback policy guard for `worker/Dockerfile.*`.

Background: several Dockerfiles used to wrap their `pip install`
lines for the SLAM / gsplat backend source builds in
`(... || echo "build failed; falls back to simulated")`. The
intent was "degrade gracefully if a CUDA build breaks", but the
result was an image that booted fine and produced silent
simulated-backend output at runtime — which the user has been
explicit about hating ("simulated results are USELESS").

PR #51 fixed this for MonoGS submodules. PR for `fix/monogs-headless-
gui-stub` extended it to MASt3R-SLAM / DROID / DPVO. This test pins
the policy so a future backend addition can't sneak the anti-pattern
back in: any `pip install` (or `python setup.py install`) wrapped in
a `|| echo`, `|| true`, or `|| :` short-circuit fails the suite.

The check is intentionally string-based + line-scoped: we don't try
to parse the shell. It catches the exact pattern that bit us.
Legitimate uses of `||` outside install lines (e.g. fallback for an
optional `apt-get` package) are not flagged."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DOCKERFILE_GLOB = "worker/Dockerfile*"

# Substrings that, when paired with a `||` short-circuit on the same
# logical RUN line, indicate a swallowed install. We match against the
# joined-up line (continuation `\` chars stripped) so multi-line RUNs
# are caught.
_INSTALL_NEEDLES = (
    "pip install",
    "setup.py install",
    "setup.py develop",
    "python -m pip install",
    "python3 -m pip install",
)
_FALLBACK_PATTERNS = (
    re.compile(r"\|\|\s*echo\b"),
    re.compile(r"\|\|\s*true\b"),
    re.compile(r"\|\|\s*:\s*$"),
    re.compile(r"\|\|\s*:\s*&&"),
)


def _iter_run_lines(dockerfile: Path):
    """Yield (line_no, joined_line) for every logical RUN command in
    the dockerfile, joining `\\`-continued lines into one string."""
    text = dockerfile.read_text(encoding="utf-8")
    raw_lines = text.splitlines()
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        stripped = line.lstrip()
        if not stripped.startswith("RUN "):
            i += 1
            continue
        start = i + 1
        joined = stripped[4:]
        # Walk continuations.
        while joined.rstrip().endswith("\\"):
            joined = joined.rstrip()[:-1].rstrip() + " "
            i += 1
            if i >= len(raw_lines):
                break
            joined += raw_lines[i].lstrip()
        yield (start, joined)
        i += 1


def test_no_fallback_swallow_in_install_lines():
    dockerfiles = sorted(_REPO_ROOT.glob(_DOCKERFILE_GLOB))
    assert dockerfiles, (
        f"no Dockerfiles matched {_DOCKERFILE_GLOB} under {_REPO_ROOT}; "
        "test config is wrong"
    )

    offenses: list[str] = []
    for df in dockerfiles:
        for line_no, joined in _iter_run_lines(df):
            if not any(needle in joined for needle in _INSTALL_NEEDLES):
                continue
            for pat in _FALLBACK_PATTERNS:
                if pat.search(joined):
                    offenses.append(
                        f"{df.relative_to(_REPO_ROOT)}:{line_no}: "
                        f"install line guards a build failure with `{pat.pattern}` — "
                        f"strict-no-fallback policy forbids this. The whole "
                        f"image must fail loudly when an install breaks. "
                        f"See PR #51 (MonoGS submodules) for context. "
                        f"Joined RUN: {joined[:200]}"
                    )

    assert not offenses, "Dockerfile install fallbacks detected:\n" + "\n".join(
        offenses
    )

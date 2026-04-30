"""Byte-layout tests for the antimatter15/OpenSplat compressed splat
exporter (`app.processors.gsplat.export.write_compressed_splat`).

The compressed format is wire-fixed: every viewer that reads it
(antimatter15/splat, sparkjs.dev, OpenSplat resume) infers gaussian
count from `byteLength / 32` and decodes each row inline. So the only
correctness test we need is "encode known gaussians, parse the bytes
back, assert each field round-trips". No image-quality regression
needed — that lives at the trainer-level integration test.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pytest


def _decode_row(buf: bytes, idx: int) -> dict:
    """Decode one 32-byte row at `buf[idx*32 : (idx+1)*32]` using the
    same little-endian / uint8 conventions the antimatter15 viewer
    applies on load."""
    base = idx * 32
    pos = struct.unpack_from("<fff", buf, base)
    scale = struct.unpack_from("<fff", buf, base + 12)
    rgba = struct.unpack_from("BBBB", buf, base + 24)
    rot_bytes = struct.unpack_from("BBBB", buf, base + 28)
    # Viewer reverses with `(byte - 128) / 128`.
    rot = tuple((b - 128) / 128.0 for b in rot_bytes)
    return {
        "pos": pos,
        "scale": scale,
        "rgba": rgba,
        "rot": rot,
    }


def test_empty_splat_writes_zero_bytes(tmp_path: Path) -> None:
    """Zero gaussians → zero-byte file. The viewer's `byteLength / 32`
    count check should produce 0 splats and not error."""
    from app.processors.gsplat.export import write_compressed_splat

    out = tmp_path / "splat.splat"
    write_compressed_splat(
        out,
        means=np.empty((0, 3), dtype=np.float32),
        colors=np.empty((0, 3), dtype=np.float32),
        opacities=np.empty((0,), dtype=np.float32),
        scales=np.empty((0, 3), dtype=np.float32),
        rotations=np.empty((0, 4), dtype=np.float32),
    )
    assert out.exists()
    assert out.stat().st_size == 0


def test_single_gaussian_round_trips(tmp_path: Path) -> None:
    """One known gaussian: position, scale-after-exp, RGB, opacity-
    after-sigmoid, and quaternion all decode within the precision the
    format allows (uint8 quantisation for color/opacity/rotation)."""
    from app.processors.gsplat.export import write_compressed_splat

    out = tmp_path / "splat.splat"
    write_compressed_splat(
        out,
        means=np.array([[1.5, -2.0, 0.25]], dtype=np.float32),
        colors=np.array([[1.0, 0.5, 0.0]], dtype=np.float32),
        # opacity logit -> sigmoid(0) = 0.5 -> ~127 byte
        opacities=np.array([0.0], dtype=np.float32),
        # log-space scale; exp(log(0.1)) = 0.1
        scales=np.array([[math.log(0.1), math.log(0.05), math.log(0.02)]], dtype=np.float32),
        # identity quaternion wxyz; encodes to (255, 128, 128, 128) post mapping
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )

    buf = out.read_bytes()
    assert len(buf) == 32
    row = _decode_row(buf, 0)

    # Position is a straight float32 round-trip.
    assert row["pos"] == pytest.approx((1.5, -2.0, 0.25), rel=0, abs=1e-6)
    # Scale is post-exp from log-space.
    assert row["scale"] == pytest.approx((0.1, 0.05, 0.02), rel=0, abs=1e-6)
    # RGB color: 1.0 -> 255, 0.5 -> 127 (round-down via cast), 0.0 -> 0.
    assert row["rgba"][0] == 255
    assert row["rgba"][1] in (127, 128)  # rounding mode tolerance
    assert row["rgba"][2] == 0
    # Opacity sigmoid(0) = 0.5 → 127.5 → byte 127 or 128 (rounding).
    assert row["rgba"][3] in (127, 128)
    # Identity quaternion wxyz: w=1, others=0 → bytes (255, 128, 128, 128)
    # which decode to (1, 0, 0, 0) within ±1/128 tolerance.
    assert row["rot"][0] == pytest.approx(1.0, abs=1 / 128)
    assert row["rot"][1] == pytest.approx(0.0, abs=1 / 128)
    assert row["rot"][2] == pytest.approx(0.0, abs=1 / 128)
    assert row["rot"][3] == pytest.approx(0.0, abs=1 / 128)


def test_total_size_is_n_times_32(tmp_path: Path) -> None:
    """File size = N * 32 — the contract antimatter15/OpenSplat readers
    rely on to count gaussians."""
    from app.processors.gsplat.export import write_compressed_splat

    out = tmp_path / "splat.splat"
    n = 137
    rng = np.random.default_rng(7)
    write_compressed_splat(
        out,
        means=rng.normal(size=(n, 3)).astype(np.float32),
        colors=rng.uniform(0, 1, size=(n, 3)).astype(np.float32),
        opacities=rng.normal(size=n).astype(np.float32),
        scales=rng.normal(scale=0.5, size=(n, 3)).astype(np.float32),
        rotations=rng.normal(size=(n, 4)).astype(np.float32),
    )
    assert out.stat().st_size == n * 32


def test_zero_norm_quaternion_doesnt_crash(tmp_path: Path) -> None:
    """A zero quaternion is degenerate but mustn't divide-by-zero —
    real trainer outputs occasionally contain near-zero rows from
    pruning races, and we don't want the export to raise."""
    from app.processors.gsplat.export import write_compressed_splat

    out = tmp_path / "splat.splat"
    write_compressed_splat(
        out,
        means=np.zeros((1, 3), dtype=np.float32),
        colors=np.zeros((1, 3), dtype=np.float32),
        opacities=np.zeros((1,), dtype=np.float32),
        scales=np.zeros((1, 3), dtype=np.float32),
        rotations=np.zeros((1, 4), dtype=np.float32),
    )
    assert out.stat().st_size == 32
    row = _decode_row(out.read_bytes(), 0)
    # All four bytes mapped from 0.0 → 128 → decoded to 0.0.
    assert row["rot"] == pytest.approx((0.0, 0.0, 0.0, 0.0), abs=1 / 128)


def test_high_opacity_logits_saturate_to_255(tmp_path: Path) -> None:
    """sigmoid(very-large) ≈ 1.0 → byte 255. Confirms the clip path
    for runaway logits doesn't overflow uint8."""
    from app.processors.gsplat.export import write_compressed_splat

    out = tmp_path / "splat.splat"
    write_compressed_splat(
        out,
        means=np.zeros((1, 3), dtype=np.float32),
        colors=np.ones((1, 3), dtype=np.float32),
        opacities=np.array([50.0], dtype=np.float32),
        scales=np.zeros((1, 3), dtype=np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    row = _decode_row(out.read_bytes(), 0)
    assert row["rgba"][3] == 255

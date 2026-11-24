"""Task 1.6: READ (SPEC 7.5)."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
from fixtures import CFG, GID, Dial, gauge_bundle
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from knotwork.knotcore.contracts import ErrorCode
from knotwork.knotcore.gauge import read
from knotwork.knotcore.io import KernelFault, load_bundle

SCALES = {
    "linear": dict(labels=("00", "20", "40", "60", "80"), vmin=0.0, vmax=80.0, zones=(40.0, 60.0)),
    "offset": dict(labels=("10", "30", "50", "70", "90"), vmin=10.0, vmax=90.0, zones=(50.0, 70.0)),
}


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    scale=st.sampled_from(sorted(SCALES)),
    major=st.integers(min_value=0, max_value=3),
    minor=st.integers(min_value=4, max_value=16),
)
def test_read_recovers_integer_values(scale: str, major: int, minor: int) -> None:
    """Integer values at least 4 units from a tick (the needle would cover that tick's label)."""
    sc = SCALES[scale]
    value = float(sc["vmin"] + 20 * major + minor)  # type: ignore[operator]
    with tempfile.TemporaryDirectory() as tmp:
        b = load_bundle(gauge_bundle(Path(tmp), Dial(value=value, **sc)))  # type: ignore[arg-type]
        (v, tol), ev = read(b, GID, CFG)
    span = sc["vmax"] - sc["vmin"]  # type: ignore[operator]
    assert abs(v - value) <= tol, (v, value, tol)
    assert tol < 0.01 * span
    assert ev["axis_offset"] < 2.0


def test_displaced_pivot_is_offcenter(tmp_path: Path) -> None:
    # ~15 px perpendicular to the needle at value 47
    b = load_bundle(gauge_bundle(tmp_path, Dial(pivot_offset=(13.8, 6.0))))
    with pytest.raises(KernelFault) as e:
        read(b, GID, CFG)
    assert e.value.code is ErrorCode.E_NEEDLE_OFFCENTER


def test_ticks_out_of_value_order(tmp_path: Path) -> None:
    b = load_bundle(gauge_bundle(tmp_path, Dial(tick_values=(0.0, 40.0, 20.0, 60.0, 80.0))))
    with pytest.raises(KernelFault) as e:
        read(b, GID, CFG)
    assert e.value.code is ErrorCode.E_SCALE_MISMATCH


def test_never_reads_claimed_center(gauge_dir: Path, tmp_path: Path) -> None:
    ref = read(load_bundle(gauge_dir), GID, CFG)[0]
    d = tmp_path / "moved"
    shutil.copytree(gauge_dir, d)
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    man["gauges"][0]["claimed_center"] = [100.0, 900.0]
    man["gauges"][0]["claimed_radius"] = 5.0
    (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    assert read(load_bundle(d), GID, CFG)[0] == ref

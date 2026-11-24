"""Emission mask, answer commitments, append-only burn ledger consulted by `generate --split test`."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import pytest
import yaml

from knotwork.freshness import emission, ledger
from knotwork.knotcore.seeds import sha256_hex

REPO = Path(__file__).resolve().parents[2]


def test_salt_created_once_and_reused(tmp_path: Path) -> None:
    p = tmp_path / "s" / "salt"
    a = emission.load_salt(p)
    assert len(a) == 32 and p.is_file()
    assert emission.load_salt(p) == a
    assert emission.mask_commit(a) == hashlib.sha256(a).hexdigest()
    p.write_bytes(b"short")
    with pytest.raises(ValueError):
        emission.load_salt(p)


def test_mask_fraction_close_to_sealed_fraction() -> None:
    salt = bytes(range(32))
    ids = [f"kw-{sha256_hex(str(i).encode())[:20]}" for i in range(5000)]
    for frac in (0.2, 0.5):
        got = sum(emission.is_sealed(salt, b, frac) for b in ids) / len(ids)
        assert abs(got - frac) <= 0.03, (frac, got)
    b = ids[0]
    want = hmac.new(salt, b.encode(), hashlib.sha256).digest()[0] / 256 < 0.2
    assert emission.is_sealed(salt, b, 0.2) is want
    assert not emission.is_sealed(salt, b, 0.0)


def test_commitment_verifies() -> None:
    salt = b"\x01" * 32
    c = emission.answer_commitment(salt, "no; 47")
    assert c == hashlib.sha256(salt + b"no; 47").hexdigest()
    assert c != emission.answer_commitment(salt, "yes; 47")
    assert c != emission.answer_commitment(b"\x02" * 32, "no; 47")


def _profile(tmp_path: Path) -> dict[str, object]:
    return {"freshness": {"burn_ledger_path": str(tmp_path / "ledger.jsonl")}}


def test_ledger_append_only(tmp_path: Path) -> None:
    prof = _profile(tmp_path)
    assert ledger.read(prof) == []
    row = {f: f"v-{f}" for f in ledger.FIELDS}
    ledger.append(prof, [row])
    first = ledger.ledger_path(prof).read_bytes()
    ledger.append(prof, [dict(row, scene_id="s2")])
    assert ledger.ledger_path(prof).read_bytes().startswith(first)
    assert [r["scene_id"] for r in ledger.read(prof)] == ["v-scene_id", "s2"]
    assert ledger.burned_scene_ids(prof) == {"v-scene_id", "s2"}
    with pytest.raises(ValueError):
        ledger.append(prof, [{"bundle_id": "x"}])


def test_generate_test_split_skips_burned_scenes(tmp_path: Path) -> None:
    from knotwork.pipeline.generate import run_generate

    prof = yaml.safe_load((REPO / "configs/default_profile.yaml").read_text(encoding="utf-8"))
    prof["freshness"]["burn_ledger_path"] = str(tmp_path / "burn.jsonl")
    ppath = tmp_path / "profile.yaml"
    ppath.write_text(yaml.safe_dump(prof), encoding="utf-8")

    def scene(out: Path) -> str:
        assert run_generate("gauge", 1, out, 3, "offline", ppath, "test", False, None, 1) == 0
        return str(json.loads((out / "index.jsonl").read_text().splitlines()[0])["scene_id"])

    first = scene(tmp_path / "a")
    assert scene(tmp_path / "b") == first  # deterministic without a ledger entry
    ledger.append(
        prof,
        [
            {
                "bundle_id": "kw-x",
                "scene_id": first,
                "slot": "vlm_a",
                "model": "m",
                "request_sha256": "r",
            }
        ],
    )
    assert scene(tmp_path / "c") != first
    # the dev split ignores the ledger
    assert run_generate("gauge", 1, tmp_path / "d", 3, "offline", ppath, "dev", False, None, 1) == 0

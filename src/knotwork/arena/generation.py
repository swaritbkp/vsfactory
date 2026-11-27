"""Dataset generation for the arena: one certified item per (block, index), with render and kernel
verification timed separately.

This follows `knotwork.pipeline.certify.run_candidate` step for step (render, claims, D1/D2, kernel
certify, content-addressed bundle id) so an arena item is exactly what the factory produces; the
only addition is the clock around the render and around the kernel.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any

from knotwork.arena.schemas import ItemRecord
from knotwork.knotcore import engine
from knotwork.knotcore.contracts import Family
from knotwork.knotcore.seeds import SeedTree
from knotwork.pipeline.certify import PROVISIONAL_ID, Candidate, emitted_answer
from knotwork.pipeline.io import bundle_id_for, file_digests, write_claims, write_json
from knotwork.plugins.registry import get_plugin
from knotwork.studio import factory
from knotwork.studio.present import AUTO, PRESETS

ATTEMPTS = 8  # scene resamples while a plugin finds no legible layout
CERTIFY_ATTEMPTS = 3  # kernel runs per item before it is recorded as quarantined


def preset_spec(preset: str) -> Any:
    """The DifficultySpec of a Studio preset (None for a preset the Studio does not know)."""
    knobs = PRESETS.get(preset)
    if not knobs:
        return None
    return factory.difficulty_from({k: v for k, v in knobs.items() if v != AUTO})


def family_for(domain: str, index: int) -> str:
    """Question families take turns within a block, so every family is covered evenly."""
    families = factory.families_of(domain)
    return families[index % len(families)]


def make_item(
    block_id: str,
    domain: str,
    preset: str,
    index: int,
    seed: int,
    data_dir: Path,
    certify_attempts: int = CERTIFY_ATTEMPTS,
) -> ItemRecord:
    """Render, certify and write one item under `<data_dir>/<block_id>/<index>/`."""
    plugin = get_plugin(domain)
    family = family_for(domain, index)
    prof = dict(factory.profile(), _family=family)
    spec = preset_spec(preset)
    takes_dials = "difficulty" in inspect.signature(plugin.sample_scene).parameters
    out = data_dir / block_id / f"{index:05d}"
    gen_ms = verify_ms = 0.0
    runs, last = 0, None
    for attempt in range(ATTEMPTS):
        at = SeedTree(seed).child("arena", domain, family, attempt)
        t0 = time.perf_counter()
        try:
            kwargs = {"difficulty": spec} if takes_dials and spec is not None else {}
            scene = plugin.sample_scene(at.child("scene"), prof, **kwargs)
            question = plugin.questions(scene, at.child("question"), Family(family))
        except RuntimeError:
            gen_ms += (time.perf_counter() - t0) * 1000
            continue
        directory = out / f"a{attempt}"
        directory.mkdir(parents=True, exist_ok=True)
        plugin.render(scene, directory, None)
        write_claims(directory, plugin.claims(scene))
        d1 = plugin.symbolic_answer(scene, question)
        d2 = plugin.geometric_answer(scene, question)
        t1 = time.perf_counter()
        gen_ms += (t1 - t0) * 1000
        cert = engine.certify(directory, question, d1, d2, prof["kernel"], PROVISIONAL_ID)
        verify_ms += (time.perf_counter() - t1) * 1000
        answer = emitted_answer(d1, cert.d3, question, prof)
        files = file_digests(directory)
        bid = bundle_id_for(files, question, answer)
        cert = cert.model_copy(update={"bundle_id": bid})
        last = Candidate(directory, scene, question, d1, d2, answer, bid, cert, files)
        runs += 1
        if last.certified or runs >= certify_attempts:
            break
    base: dict[str, Any] = {"block_id": block_id, "item_index": index, "seed": seed, "domain": domain,
            "preset": preset, "family": family, "attempts": max(runs, 1),
            "gen_ms": round(gen_ms, 1), "verify_ms": round(verify_ms, 1)}  # fmt: skip
    if last is None:
        return ItemRecord(**base, certified=False, codes=["E_NO_LEGIBLE_SCENE"])
    codes = [str(c) for c in last.certificate.model_dump(mode="json")["codes"]]
    if not last.certified:
        return ItemRecord(**base, certified=False, codes=codes, bundle_id=last.bundle_id)
    factory._write_task(last, seed, prof)  # a complete bundle: `knot verify` accepts it
    write_json(last.dir / "certificate.json", last.certificate.model_dump(mode="json"))
    write_json(last.dir / "scene.json", last.scene.model_dump(mode="json"))  # for twin suites
    return ItemRecord(
        **base, certified=True, bundle_id=last.bundle_id,
        dir=last.dir.relative_to(data_dir).as_posix(), question=last.question.text,
        options=last.question.options, answer=last.answer.model_dump(mode="json"),
        truth=last.answer.canonical, codes=codes,
    )  # fmt: skip


def make_item_job(job: dict[str, Any]) -> dict[str, Any]:
    """Process-pool entry point: a job dict in, an ItemRecord as JSON out."""
    item = make_item(job["block_id"], job["domain"], job["preset"], int(job["index"]),
                     int(job["seed"]), Path(job["data_dir"]))  # fmt: skip
    return item.model_dump(mode="json")

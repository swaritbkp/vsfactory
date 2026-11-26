"""Tangle TaskPlugin: braid-word cable tangles (SPEC 8.1). Untrusted: the kernel re-derives every
answer from the rendered pixels and cross-checks the claims emitted here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from knotwork.knotcore.contracts import (
    Answer,
    AnswerType,
    Claim,
    ClaimSet,
    Family,
    FaultCode,
    Op,
    QuestionProgram,
    RenderManifest,
    SceneIR,
)
from knotwork.knotcore.seeds import SeedTree
from knotwork.plugins.tangle import braid, geom, layout
from knotwork.plugins.tangle.render import halo_mask, hex_rgb, render_tangle
from knotwork.render import realism
from knotwork.render.canvas import make_canvas, render_cfg

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#882255",
           "#117733", "#332288", "#000000", "#8C510A", "#1111FF"]  # fmt: skip
MONO = "#2F2F2F"
#: matte industrial backplates (light enough that every palette colour keeps >= 0.2 luminance contrast)
PLATES = [(226, 226, 220), (222, 224, 226), (230, 226, 216), (218, 222, 218)]
FORMAT = {AnswerType.LABEL: "ANSWER: <label>", AnswerType.INT: "ANSWER: <integer>"}
PROMPTS = {
    Family.T1_TRACE: "The image shows cables running from lettered terminals to numbered terminals. "
    "Follow the cable that starts at terminal {S}. At which numbered terminal does it end?",
    Family.T2_CROSS: "Follow the cable that starts at terminal {S} along its whole length. "
    "How many times does it cross another cable?",
    Family.T3_ONTOP: "Cables from terminals {A} and {B} cross {m} time(s). At crossing number "
    "{k} of these, counting along cable {A} from its start, which cable is on top?",
}
MAX_LAYOUT_TRIES = 400


class TwinUnavailable(RuntimeError):
    """No valid twin edit exists for this scene/question (the pipeline retries another attempt)."""


def start_tid(label: str) -> str:
    return f"tangle/terminal:start_{label}"


def _letter(x: Any) -> braid.Letter:
    return None if x is None else tuple(int(v) for v in x)


def word_of(pay: dict[str, Any]) -> list[braid.Letter]:
    return [_letter(x) for x in pay["word"]]


def lay_of(pay: dict[str, Any], col_weights: dict[int, float] | None = None) -> layout.Layout:
    return layout.build(pay["n"], word_of(pay), pay["layout"], col_weights)


def sample_geometry(rng: np.random.Generator, n: int, word: list[braid.Letter], prof: dict[str, Any], kind: str,
                    stroke: float) -> dict[str, Any] | None:  # fmt: skip
    """Layout params for a word, or None if no legible layout exists (caller resamples)."""
    ncols = len(braid.columns(n, word))
    params = layout.plan(n, ncols, kind, stroke) or layout.plan(n, ncols, "strip", stroke)
    if params is None:
        return None
    amp = float(rng.uniform(*prof["jitter_amp_px"]))
    params |= {"jitter_amps": (rng.dirichlet([1.0, 1.0, 1.0]) * amp).tolist(),
               "jitter_freqs": [1.0, 2.0, 3.0], "jitter_phases": rng.uniform(0, 2 * math.pi, 3).tolist()}  # fmt: skip
    return params


def shuffled_slot(salt: int | None, key: str, index: int, m: int) -> int:
    """``index mod m`` seen through a run-seeded permutation of each block of ``m`` consecutive
    indices: every block still uses every slot exactly once, but which item gets which slot no
    longer follows the generation order (SPEC D-39)."""
    if salt is None:
        return index % m
    block, r = divmod(index, m)
    return int(SeedTree(int(salt)).child("balance", key, m, block).rng().permutation(m)[r])


def _pick(rng: np.random.Generator, weights: dict[str, float]) -> str:
    keys = sorted(weights)
    p = np.array([weights[k] for k in keys], dtype=np.float64)
    return str(keys[int(rng.choice(len(keys), p=p / p.sum()))])


class TanglePlugin:
    name = "tangle"
    version = "1.0.0"
    families = (Family.T1_TRACE, Family.T2_CROSS, Family.T3_ONTOP)

    # ------------------------------------------------------------------ scene
    def sample_scene(self, seeds: SeedTree, profile: dict[str, Any]) -> SceneIR:
        """Sample a braid scene. ``profile["_family"]`` (optional) conditions T2 on a target count."""
        prof, rng = profile["tangle"], seeds.rng()
        family = profile.get("_family")
        target = int(rng.integers(prof["t2_target_count"][0], prof["t2_target_count"][1] + 1))
        o = profile.get("_ordinal")
        lo_n, hi_n = int(prof["strands"][0]), int(prof["strands"][1])
        span_n = hi_n - lo_n + 1
        if o is not None and family == Family.T2_CROSS.value:  # exact T2 balance (D-34)
            t_lo, t_hi = prof["t2_target_count"]
            target = int(t_lo) + int(o) % (int(t_hi) - int(t_lo) + 1)
        regimes = prof["color_regimes"]
        if profile.get("_split") in ("test", "sealed"):
            regimes = {r: 1.0 for r in prof["benchmark_color_regimes"]}
        for _ in range(MAX_LAYOUT_TRIES):
            n = int(rng.integers(lo_n, hi_n + 1))
            if o is not None and family == Family.T1_TRACE.value:
                n = lo_n + int(o) % span_n  # n cycles; the answer position cycles within each n
            perm, word = braid.sample_word(n, rng, tuple(prof["braid_length"]), tuple(prof["r2_distractor_pairs"]),
                                           tuple(prof["identity_slots"]))  # fmt: skip
            counts = braid.crossing_counts(n, word)
            if family == Family.T2_CROSS.value and target not in counts:
                continue
            stroke = float(rng.integers(prof["stroke_px"][0], prof["stroke_px"][1] + 1))
            params = sample_geometry(rng, n, word, prof, _pick(rng, prof["layouts"]), stroke)
            if params is None:
                continue
            pay: dict[str, Any] = {"n": n, "word": [list(x) if x else None for x in word], "perm": perm, "stroke": stroke,
                   "layout": params, "style": _pick(rng, prof["styles"]), "t2_target": target,
                   "render": render_cfg(profile), "surface": realism.sample_surface(rng, "powder")}  # fmt: skip
            pay["render"]["background_rgb"] = list(PLATES[int(rng.integers(len(PLATES)))])
            if geom.validate(lay_of(pay), n, stroke) is not None:
                continue
            letters = list(prof["start_labels"])
            pay["start_labels"] = [
                letters[int(i)] for i in rng.choice(len(letters), size=n, replace=False)
            ]
            pay["end_labels"] = [str(j + 1) for j in range(n)]
            regime = _pick(rng, regimes)
            pal = [PALETTE[int(i)] for i in rng.permutation(len(PALETTE))]
            if regime == "unique":
                cols = pal[:n]
            elif regime == "partial":
                sub = pal[: math.ceil(n / 3)]
                cols = [sub[int(rng.integers(len(sub)))] for _ in range(n)]
            else:
                cols = [MONO] * n
            pay["colors"], pay["color_regime"] = [list(hex_rgb(h)) for h in cols], regime
            pay["relief"] = realism.relief_panels(rng)
            if o is not None:
                salt = profile.get("_balance_seed")
                if family == Family.T1_TRACE.value:
                    pay["answer_slot"] = shuffled_slot(salt, f"T1n{n}", int(o) // span_n, n)
                else:
                    pay["answer_slot"] = shuffled_slot(salt, str(family), int(o), 2)
            return SceneIR(plugin=self.name, plugin_version=self.version, seed_path=seeds.path_str,
                           params={"n_strands": n, "layout": params["kind"], "style": pay["style"],
                                   "color_regime": regime}, payload=pay)  # fmt: skip
        raise RuntimeError(
            f"no legible tangle layout in {MAX_LAYOUT_TRIES} tries ({seeds.path_str})"
        )

    def render(
        self, scene: SceneIR, out_dir: Path, fault: FaultCode | None = None
    ) -> RenderManifest:
        pay = scene.payload
        c = make_canvas(pay["render"])
        weights = None
        if fault is FaultCode.M4:
            weights = m4_weights(pay)
        lay = lay_of(pay, weights)
        realism.draw_relief(c, pay.get("relief", []))
        realism.corner_screws(c, "tangle")
        terms = render_tangle(c, pay, lay.paths, lay.crossings, fault)
        crossings = lay.crossings
        realism.add_surface(c, pay["surface"], ("label",), lambda cv: halo_mask(crossings, cv.size))
        return c.finish(out_dir, self.name, terminals=terms)

    # ------------------------------------------------------------------ claims / questions
    def claims(self, scene: SceneIR) -> ClaimSet:
        pay = scene.payload
        n, word, labels = pay["n"], word_of(pay), pay["start_labels"]
        ends = braid.end_position(n, word)
        counts = braid.crossing_counts(n, word)
        out: list[Claim] = []
        for s in range(n):
            end = pay["end_labels"][ends[s]]
            out.append(
                Claim(
                    claim_id=f"trace_{labels[s]}",
                    op=Op.TRACE,
                    args=[start_tid(labels[s])],
                    expected=_label(end),
                )
            )
            out.append(Claim(claim_id=f"cross_{labels[s]}", op=Op.CROSS, args=[start_tid(labels[s]), "*"],
                             expected=_int(counts[s])))  # fmt: skip
        for a in range(n):
            for b in range(a + 1, n):
                for k, x in enumerate(braid.pair_crossings(n, word, a, b)):
                    out.append(Claim(claim_id=f"ontop_{labels[a]}{labels[b]}_{k}", op=Op.ONTOP,
                                     args=[start_tid(labels[a]), start_tid(labels[b]), k],
                                     expected=_label(labels[x.over])))  # fmt: skip
        return ClaimSet(plugin=self.name, claims=out)

    def questions(self, scene: SceneIR, seeds: SeedTree, family: Family) -> QuestionProgram:
        pay, rng = scene.payload, seeds.rng()
        n, word, labels = pay["n"], word_of(pay), pay["start_labels"]
        pid = f"tangle/{family.value}/{seeds.digest()[:16]}"
        if family is Family.T1_TRACE:
            s = int(rng.integers(n))
            if "answer_slot" in pay:  # the strand ending at the balanced answer position (D-34)
                ends = braid.end_position(n, word)
                s = next(x for x in range(n) if ends[x] == int(pay["answer_slot"]) % n)
            text = PROMPTS[family].format(S=labels[s])
            steps: list[dict[str, Any]] = [
                {"step_id": "s0", "op": "TRACE", "args": [start_tid(labels[s])]}
            ]
            return _program(
                pid, family, steps, text, AnswerType.LABEL, list(pay["end_labels"]), {"n": n}
            )
        if family is Family.T2_CROSS:
            counts = braid.crossing_counts(n, word)
            cands = [s for s in range(n) if counts[s] == pay["t2_target"]] or list(range(n))
            s = cands[int(rng.integers(len(cands)))]
            text = PROMPTS[family].format(S=labels[s])
            steps = [{"step_id": "s0", "op": "CROSS", "args": [start_tid(labels[s]), "*"]}]
            return _program(
                pid, family, steps, text, AnswerType.INT, None, {"n": n, "count": counts[s]}
            )
        pairs = [
            (a, b) for a in range(n) for b in range(a + 1, n) if braid.pair_crossings(n, word, a, b)
        ]
        if not pairs:
            raise TwinUnavailable("no crossing pair for T3")
        a, b = pairs[int(rng.integers(len(pairs)))]
        if rng.random() < 0.5:
            a, b = b, a
        m = len(braid.pair_crossings(n, word, a, b))
        k = int(rng.integers(m))
        if "answer_slot" in pay:  # list the top strand at the balanced option index (D-34)
            top = _ordered(n, word, a, b)[k].over
            if (top == a) != (int(pay["answer_slot"]) % 2 == 0):
                a, b = b, a  # crossings are in word order for either strand order: k is unchanged
        text = PROMPTS[family].format(A=labels[a], B=labels[b], m=m, k=k + 1)
        steps = [
            {
                "step_id": "s0",
                "op": "ONTOP",
                "args": [start_tid(labels[a]), start_tid(labels[b]), k],
            }
        ]
        return _program(
            pid, family, steps, text, AnswerType.LABEL, [labels[a], labels[b]], {"n": n, "m": m}
        )

    # ------------------------------------------------------------------ answers
    def symbolic_answer(self, scene: SceneIR, program: QuestionProgram) -> Answer:
        pay = scene.payload
        n, word = pay["n"], word_of(pay)
        args = _strand_args(pay, program)
        if program.family is Family.T1_TRACE:
            return _label(pay["end_labels"][braid.end_position(n, word)[args[0]]])
        if program.family is Family.T2_CROSS:
            return _int(braid.crossing_counts(n, word)[args[0]])
        a, b, k = args
        return _label(pay["start_labels"][_ordered(n, word, a, b)[k].over])

    def geometric_answer(self, scene: SceneIR, program: QuestionProgram) -> Answer | None:
        pay = scene.payload
        lay = lay_of(pay)
        args = _strand_args(pay, program)
        ends = [lay.paths[pay["perm"][j]][-1] for j in range(pay["n"])]
        got = geom.geometric_answer(lay, program.family.value, args, ends)
        if program.family is Family.T1_TRACE:
            return _label(pay["end_labels"][got])
        if program.family is Family.T2_CROSS:
            return _int(got)
        return _label(pay["start_labels"][got])

    # ------------------------------------------------------------------ twins
    def causal_twin(
        self, scene: SceneIR, program: QuestionProgram, seeds: SeedTree
    ) -> tuple[SceneIR, str]:
        pay, rng = _copy(scene.payload), seeds.rng()
        n, word = pay["n"], word_of(pay)
        args = _strand_args(pay, program)
        if program.family is Family.T1_TRACE:
            j = braid.end_position(n, word)[args[0]]
            other = [k for k in range(n) if k != j][int(rng.integers(n - 1))]
            ends = pay["end_labels"]
            ends[j], ends[other] = ends[other], ends[j]
            return _scene(scene, pay), f"swap end labels {ends[other]}<->{ends[j]}"
        if program.family is Family.T2_CROSS:
            slots = _slots(n, word, lambda pos, i: args[0] in (pos[i], pos[i + 1]), rng)
            for k, i in slots:
                eps = int(rng.choice([1, -1]))
                trial = _copy(pay)
                trial["word"] = _fill(pay["word"], k, i, eps)
                if geom.validate(lay_of(trial), n, float(pay["stroke"])) is None:
                    return _scene(scene, trial), f"R-II pair on ID slot {k} at position {i}"
            raise TwinUnavailable("no ID slot next to the query strand")
        a, b, k = args
        target = _ordered(n, word, a, b)[k]
        letter = list(pay["word"][target.letter])
        letter[1] = -letter[1]
        pay["word"][target.letter] = letter
        return _scene(scene, pay), f"flip crossing {k + 1} of the queried pair"

    def nuisance_twin(self, scene: SceneIR, program: QuestionProgram, seeds: SeedTree) -> SceneIR:
        pay, rng = _copy(scene.payload), seeds.rng()
        n = pay["n"]
        involved = set(
            _strand_args(pay, program)[:2]
            if program.family is Family.T3_ONTOP
            else _strand_args(pay, program)[:1]
        )
        for _ in range(20):
            trial = _copy(pay)
            amp = max(8.0, float(sum(trial["layout"]["jitter_amps"])))
            trial["layout"]["jitter_amps"] = (rng.dirichlet([1.0, 1.0, 1.0]) * amp).tolist()
            trial["layout"]["jitter_phases"] = rng.uniform(0, 2 * math.pi, 3).tolist()
            trial["style"] = "harness" if pay["style"] == "schematic" else "schematic"
            trial["surface"] = realism.sample_surface(rng, "powder")  # new lighting / grime / grain
            plates = [p for p in PLATES if list(p) != pay["render"]["background_rgb"]]
            trial["render"]["background_rgb"] = list(plates[int(rng.integers(len(plates)))])
            if rng.random() < 0.5:
                slots = _slots(
                    n, word_of(trial), lambda pos, i: not {pos[i], pos[i + 1]} & involved, rng
                )
                if slots:
                    k, i = slots[0]
                    trial["word"] = _fill(trial["word"], k, i, int(rng.choice([1, -1])))
            if geom.validate(lay_of(trial), n, float(pay["stroke"])) is None:
                return _scene(scene, trial)
        raise TwinUnavailable("no valid nuisance layout")


# ---------------------------------------------------------------------- helpers
def _label(v: str) -> Answer:
    return Answer(type=AnswerType.LABEL, value=v, canonical=v)


def _int(v: int) -> Answer:
    return Answer(type=AnswerType.INT, value=int(v), canonical=str(int(v)))


def _program(pid: str, family: Family, steps: list[dict[str, Any]], text: str, at: AnswerType,
             options: list[str] | None, difficulty: dict[str, float]) -> QuestionProgram:  # fmt: skip
    full = f"{text}\nEnd your reply with a final line of the form '{FORMAT[at]}'."
    return QuestionProgram(program_id=pid, family=family, steps=steps, text=full, answer_type=at,  # type: ignore[arg-type]
                           options=options, difficulty={k: float(v) for k, v in difficulty.items()})  # fmt: skip


def _strand_args(pay: dict[str, Any], program: QuestionProgram) -> list[int]:
    labels = pay["start_labels"]
    out = []
    for a in program.steps[0].args:
        if isinstance(a, str) and a.startswith("tangle/terminal:start_"):
            out.append(labels.index(a.rsplit("_", 1)[1]))
        elif isinstance(a, int) and not isinstance(a, bool):
            out.append(a)
    return out


def _ordered(n: int, word: list[braid.Letter], a: int, b: int) -> list[braid.BraidCrossing]:
    """Crossings of (a, b) in order along strand a (left to right = word order)."""
    return braid.pair_crossings(n, word, a, b)


def _copy(pay: dict[str, Any]) -> dict[str, Any]:
    import copy

    return copy.deepcopy(pay)


def _scene(scene: SceneIR, pay: dict[str, Any]) -> SceneIR:
    return SceneIR(plugin=scene.plugin, plugin_version=scene.plugin_version, seed_path=scene.seed_path,
                   params=dict(scene.params), payload=pay)  # fmt: skip


def _slots(
    n: int, word: list[braid.Letter], ok: Any, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """(word index, generator i) of empty ID slots where ``ok(arrangement, i)``; shuffled."""
    out: list[tuple[int, int]] = []
    pos = list(range(n))
    for k, letter in enumerate(word):
        if letter is None:
            out.extend((k, i) for i in range(n - 1) if ok(pos, i))
        else:
            i = letter[0]
            pos[i], pos[i + 1] = pos[i + 1], pos[i]
    return [out[int(j)] for j in rng.permutation(len(out))]


def _fill(word: list[Any], k: int, i: int, eps: int) -> list[Any]:
    """Replace the ID slot at ``k`` by an R-II pair (i, eps)(i, -eps) occupying its two columns."""
    return [*word[:k], [i, eps, 1], [i, -eps, 1], *word[k + 1 :]]


def m4_crossing(pay: dict[str, Any]) -> dict[str, Any]:
    lay = lay_of(pay)
    return lay.crossings[len(lay.crossings) // 2]


def m4_weights(pay: dict[str, Any]) -> dict[int, float]:
    """M4: widen the chosen crossing's column by max(6, k), k bringing its nominal S-curve crossing
    angle to ~16 deg (each strand within ~8 deg of horizontal: (pi/2) dy / w' = tan 8 deg; D-17)."""
    lay = lay_of(pay)
    ncols = max(lay.n_cols, 1)
    w = (pay["layout"]["x1"] - pay["layout"]["x0"]) / ncols
    need = (math.pi / 2) * float(pay["layout"]["dy"]) / math.tan(math.radians(8.0)) / w
    k = max(6.0, need)
    # widening one column to k * w' with the total width fixed: solve k * w' = target for weight
    weight = k * (ncols - 1) / max(ncols - k, 1.0) if k < ncols else k * ncols
    return {int(m4_crossing(pay)["col"]): float(weight)}


def fault_effective(scene: SceneIR, fault: FaultCode) -> bool:
    pay = scene.payload
    if fault is FaultCode.M2:
        return any(c.over != max(c.a, c.b) for c in braid.compose(pay["n"], word_of(pay))[1])
    if fault in (FaultCode.M4, FaultCode.M6) and not braid.compose(pay["n"], word_of(pay))[1]:
        return False
    if fault is FaultCode.M4:
        from shapely.geometry import LineString, Point

        x = m4_crossing(pay)
        lay = lay_of(pay, m4_weights(pay))
        near = min(
            lay.crossings, key=lambda c: abs(c["col"] - x["col"]) + abs(c["letter"] - x["letter"])
        )
        la, lb = LineString(lay.paths[near["a"]]), LineString(lay.paths[near["b"]])
        return geom.crossing_angle(la, lb, Point(near["center"])) < 20.0
    return fault in (
        FaultCode.M1,
        FaultCode.M3,
        FaultCode.M5,
        FaultCode.M6,
        FaultCode.M7,
        FaultCode.M10,
    )


PLUGIN = TanglePlugin()

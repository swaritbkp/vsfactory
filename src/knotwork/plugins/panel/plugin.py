"""Panel TaskPlugin: lettered sockets cabled through a tangle to labelled gauges (SPEC 8.3).

Composition reuses the tangle braid/layout/render helpers and the gauge dial helpers on one canvas.
Untrusted; D1 comes from the symbolic composition, the kernel executes the programs on pixels.
"""

from __future__ import annotations

import copy
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
    PixelBox,
    PremiseValue,
    QuestionProgram,
    RenderManifest,
    SceneIR,
    TerminalEntry,
)
from knotwork.knotcore.seeds import SeedTree
from knotwork.plugins.gauge.plugin import (
    BEZELS,
    TINTS,
    float_answer,
    pick_value,
    sample_dial,
    tol_units,
    values_for,
    zone_answer,
)
from knotwork.plugins.gauge.render import draw_dial
from knotwork.plugins.gauge.scale import ZONES, zone_of
from knotwork.plugins.panel import compose
from knotwork.plugins.panel.programs import PREMISE, program, socket_tid
from knotwork.plugins.tangle import braid, geom
from knotwork.plugins.tangle.layout import Layout
from knotwork.plugins.tangle.plugin import MONO, PALETTE, _pick
from knotwork.plugins.tangle.render import (
    LABEL_RGB,
    add_terminal_labels,
    draw_cables,
    hex_rgb,
    unit,
)
from knotwork.render.canvas import make_canvas, render_cfg

MAX_TRIES = 400
#: Tick labels sit nearer the hub on the small panel dials so ticks stay out of their OCR crops.
PANEL_LABEL_FRAC = 0.47


class TwinUnavailable(RuntimeError):
    """No admissible twin edit (the pipeline retries another attempt)."""


def _letter(x: Any) -> braid.Letter:
    return None if x is None else tuple(int(v) for v in x)


def strand_gauge(pay: dict[str, Any]) -> dict[int, int]:
    """socket strand -> index of the gauge its cable reaches."""
    exits, _ = braid.compose(pay["n"], [_letter(x) for x in pay["word"]])
    return {s: pay["dest_gauge"][j] for j, s in enumerate(exits) if pay["dest_gauge"][j] >= 0}


def socket_values(pay: dict[str, Any]) -> list[tuple[str, float]]:
    sg = strand_gauge(pay)
    return [
        (pay["sockets"][str(s)], float(pay["dials"][g]["value"])) for s, g in sorted(sg.items())
    ]


def target_gauge(pay: dict[str, Any]) -> dict[str, Any]:
    strand = next(int(s) for s, lab in pay["sockets"].items() if lab == pay["target"])
    return dict(pay["dials"][strand_gauge(pay)[strand]])


def _scene(scene: SceneIR, pay: dict[str, Any]) -> SceneIR:
    return SceneIR(plugin=scene.plugin, plugin_version=scene.plugin_version, seed_path=scene.seed_path,
                   params=dict(scene.params), payload=pay)  # fmt: skip


def argmax_ok(pay: dict[str, Any]) -> bool:
    vals = sorted((v for _, v in socket_values(pay)), reverse=True)
    return len(vals) < 2 or vals[0] - vals[1] >= float(pay["argmax_margin"])


def red_count(pay: dict[str, Any]) -> int:
    return sum(zone_of(float(d["value"]), tuple(d["zones"])) == "red" for d in pay["dials"])


class PanelPlugin:
    name = "panel"
    version = "1.0.0"
    families = (Family.P1_READ_TRACE, Family.P2_ZONE_TRACE, Family.P3_ARGMAX_PORT, Family.P4_COUNT_ZONE,
                Family.P5_PREMISE_CONTROL, Family.T_GASLIGHT)  # fmt: skip

    # ------------------------------------------------------------------ scene
    def sample_scene(self, seeds: SeedTree, profile: dict[str, Any]) -> SceneIR:
        prof, tprof, gprof, rng = profile["panel"], profile["tangle"], profile["gauge"], seeds.rng()
        family = profile.get("_family") or Family.P1_READ_TRACE.value
        regimes = tprof["color_regimes"]
        if profile.get("_split") in ("test", "sealed"):
            regimes = {r: 1.0 for r in tprof["benchmark_color_regimes"]}
        for _ in range(MAX_TRIES):
            k = int(rng.integers(prof["gauges"][0], prof["gauges"][1] + 1))
            e = int(rng.integers(prof["extra_strands"][0], prof["extra_strands"][1] + 1))
            n = k + e
            perm = [int(x) for x in rng.permutation(n)]
            word = braid.reduced_word(perm, rng)
            if n > 1 and rng.random() < 0.5:
                i, eps, at = (
                    int(rng.integers(n - 1)),
                    int(rng.choice([1, -1])),
                    int(rng.integers(len(word) + 1)),
                )
                word[at:at] = [(i, eps), (i, -eps)]
            stroke = (
                7.0  # 5x5 colour witness stays inside the stroke; port clearance (8 px) >= 1 stroke
            )
            plan = compose.braid_plan(n, word, stroke)
            if plan is None:
                continue
            socket_strands = sorted(int(x) for x in rng.choice(n, size=k, replace=False))
            exits, _ = braid.compose(n, word)
            pattern = [exits[j] in socket_strands for j in range(n)]
            centers = compose.dial_slots(k)
            ports = [compose.port_xy(c) for c in centers]
            pads = compose.pad_slots(rng, [p[1] for p in ports], pattern)
            if pads is None:
                continue
            dests, kinds, gidx, pi, qi = [], [], [], 0, 0
            for is_port in pattern:
                if is_port:
                    dests.append(list(ports[pi]))
                    kinds.append("port")
                    gidx.append(pi)
                    pi += 1
                else:
                    dests.append([compose.pad_x(k), pads[qi]])
                    kinds.append("pad")
                    gidx.append(-1)
                    qi += 1
            letters = list(tprof["start_labels"])
            picks = [letters[int(i)] for i in rng.choice(len(letters), size=k, replace=False)]
            dials = [
                sample_dial(
                    rng, gprof, f"g{i + 1}", centers[i], compose.DIAL_R, None, PANEL_LABEL_FRAC
                )
                for i in range(k)
            ]
            regime = _pick(rng, regimes)
            pal = [PALETTE[int(i)] for i in rng.permutation(len(PALETTE))]
            cols = pal[:n] if regime == "unique" else ([MONO] * n if regime == "mono" else
                   [pal[: max(1, (n + 2) // 3)][int(rng.integers(max(1, (n + 2) // 3)))] for _ in range(n)])  # fmt: skip
            tol = tol_units(gprof, dials[0])
            pay: dict[str, Any] = {
                "n": n, "k": k, "word": [list(x or ()) for x in word], "perm": perm, "stroke": stroke, "braid": plan,
                "style": _pick(rng, tprof["styles"]), "colors": [list(hex_rgb(h)) for h in cols], "color_regime": regime,
                "sockets": {str(s): picks[r] for r, s in enumerate(socket_strands)}, "dests": dests, "dest_kind": kinds,
                "dest_gauge": gidx, "dials": dials, "render": render_cfg(profile), "grade_tol": tol,
                "argmax_margin": float(prof["argmax_min_margin_tol"]) * tol, "target": None, "claimed": None,
            }  # fmt: skip
            if not self._condition(pay, family, rng, gprof):
                continue
            lay = Layout(
                compose.build_paths(pay)[0], compose.build_paths(pay)[1], plan["dy"], 0.0, 0
            )
            if geom.validate(lay, n, stroke) is not None:
                continue
            return SceneIR(plugin=self.name, plugin_version=self.version, seed_path=seeds.path_str,
                           params={"n_gauges": k, "n_strands": n, "style": pay["style"], "color_regime": regime},
                           payload=pay)  # fmt: skip
        raise RuntimeError(f"no valid panel layout in {MAX_TRIES} tries ({seeds.path_str})")

    def _condition(
        self, pay: dict[str, Any], family: str, rng: np.random.Generator, gprof: dict[str, Any]
    ) -> bool:
        """Adjust needle values so the family's answer distribution is balanced and well posed."""
        tol, k = float(pay["grade_tol"]), pay["k"]
        socket_labels = sorted(pay["sockets"].values())
        if family in (
            Family.P1_READ_TRACE.value,
            Family.P2_ZONE_TRACE.value,
            *(f.value for f in PREMISE),
        ):
            pay["target"] = socket_labels[int(rng.integers(len(socket_labels)))]
            if family != Family.P1_READ_TRACE.value:
                g = next(d for d in pay["dials"] if d["name"] == target_gauge(pay)["name"])
                zone = str(ZONES[int(rng.integers(3))])
                v = pick_value(rng, g, tol, zone=zone)
                if v is None:
                    return False
                g["value"] = float(v)
                if family in (f.value for f in PREMISE):
                    holds = family == Family.P5_PREMISE_CONTROL.value
                    others = [z for z in ZONES if z != zone]
                    pay["claimed"] = zone if holds else others[int(rng.integers(2))]
        elif family == Family.P3_ARGMAX_PORT.value:
            for _ in range(60):
                if argmax_ok(pay):
                    break
                for d in pay["dials"]:
                    d["value"] = float(pick_value(rng, d, tol) or d["value"])
            return argmax_ok(pay)
        elif family == Family.P4_COUNT_ZONE.value:
            want = int(rng.integers(0, k + 1))
            reds = set(int(i) for i in rng.choice(k, size=want, replace=False))
            for i, d in enumerate(pay["dials"]):
                zone = "red" if i in reds else ["green", "amber"][int(rng.integers(2))]
                v = pick_value(rng, d, tol, zone=zone)
                if v is None:
                    return False
                d["value"] = float(v)
        del gprof
        return True

    # ------------------------------------------------------------------ render
    def render(
        self, scene: SceneIR, out_dir: Path, fault: FaultCode | None = None
    ) -> RenderManifest:
        pay = scene.payload
        c = make_canvas(pay["render"])
        paths, crossings, dests = compose.build_paths(pay)
        gauges, terms = [], []
        for i, d in enumerate(pay["dials"]):
            g = draw_dial(c, "panel", d)
            cx, cy = d["center"]
            lab = c.new_object(f"panel/label:gauge_G{i + 1}", "label", LABEL_RGB, parent="panel/labels:gauges",
                               label_text=f"G{i + 1}")  # fmt: skip
            c.text(lab, (cx, cy + compose.DIAL_R + 24.0), f"G{i + 1}", 26)
            row = c.mask_array(g.dial_obj)[int(round(cy))]
            x1 = int(np.nonzero(row)[0].min()) - compose.PORT_GAP
            tid = f"panel/terminal:port_g{i + 1}"
            box = PixelBox(x0=x1 - 20, y0=int(round(cy)) - 10, x1=x1, y1=int(round(cy)) + 10)
            terms.append(TerminalEntry(terminal_id=tid, role="port", box=box, label_obj=lab))
            gauges.append(g.model_copy(update={"port_terminal": tid}))
        colors = [tuple(x) for x in pay["colors"]]
        draw_cables(c, "panel", paths, crossings, colors, float(pay["stroke"]), pay["style"])
        n = pay["n"]
        starts = [paths[s][0] for s in range(n)]
        outward = [-unit(paths[s][1] - paths[s][0]) for s in range(n)]
        names = [pay["sockets"].get(str(s), f"x{s}") for s in range(n)]
        texts = [pay["sockets"].get(str(s)) for s in range(n)]
        terms += add_terminal_labels(c, "panel", "start", starts, outward, names, texts)
        pads = [j for j, kind in enumerate(pay["dest_kind"]) if kind == "pad"]
        terms += add_terminal_labels(c, "panel", "end", [np.asarray(dests[j]) for j in pads],
                                     [np.array([1.0, 0.0])] * len(pads), [f"pad{j}" for j in pads], [None] * len(pads))  # fmt: skip
        del fault
        return c.finish(out_dir, self.name, terminals=terms, gauges=gauges)

    # ------------------------------------------------------------------ claims / questions
    def claims(self, scene: SceneIR) -> ClaimSet:
        pay = scene.payload
        n, word = pay["n"], [_letter(x) for x in pay["word"]]
        exits, _ = braid.compose(n, word)
        names = [pay["sockets"].get(str(s), f"x{s}") for s in range(n)]
        counts = braid.crossing_counts(n, word)
        out: list[Claim] = []
        for j, s in enumerate(exits):
            g = pay["dest_gauge"][j]
            end = f"G{g + 1}" if g >= 0 else f"panel/terminal:end_pad{j}"
            out.append(
                Claim(
                    claim_id=f"trace_{names[s]}",
                    op=Op.TRACE,
                    args=[socket_tid(names[s])],
                    expected=_label(end),
                )
            )
        for s in range(n):
            out.append(
                Claim(
                    claim_id=f"cross_{names[s]}",
                    op=Op.CROSS,
                    args=[socket_tid(names[s]), "*"],
                    expected=_int(counts[s]),
                )
            )
            for b in range(s + 1, n):
                for k, x in enumerate(braid.pair_crossings(n, word, s, b)):
                    out.append(Claim(claim_id=f"ontop_{names[s]}_{names[b]}_{k}", op=Op.ONTOP,
                                     args=[socket_tid(names[s]), socket_tid(names[b]), k], expected=_label(_top_label(pay, x.over))))  # fmt: skip
        for d in pay["dials"]:
            gid = f"panel/gauge:{d['name']}"
            out.append(
                Claim(
                    claim_id=f"read_{d['name']}",
                    op=Op.READ,
                    args=[gid],
                    expected=float_answer(d["value"]),
                )
            )
            out.append(Claim(claim_id=f"zone_{d['name']}", op=Op.ZONE, args=[gid],
                             expected=zone_answer(zone_of(d["value"], tuple(d["zones"])))))  # fmt: skip
        return ClaimSet(plugin=self.name, claims=out)

    def questions(self, scene: SceneIR, seeds: SeedTree, family: Family) -> QuestionProgram:
        pay = scene.payload
        holds = None
        if family in PREMISE:
            holds = (
                zone_of(target_gauge(pay)["value"], tuple(target_gauge(pay)["zones"]))
                == pay["claimed"]
            )
        span = float(pay["dials"][0]["vmax"]) - float(pay["dials"][0]["vmin"])
        return program(family, f"panel/{family.value}/{seeds.digest()[:16]}", pay["target"],
                       sorted(pay["sockets"].values()), pay["claimed"], holds, span)  # fmt: skip

    def twin_question(self, scene: SceneIR, base: QuestionProgram) -> QuestionProgram:
        """Twin's question: same program; premise truth (and so the family) re-derived from the twin."""
        pay = scene.payload
        fam = base.family
        holds = None
        if fam in PREMISE:
            g = target_gauge(pay)
            holds = zone_of(g["value"], tuple(g["zones"])) == pay["claimed"]
        return program(fam, base.program_id, pay["target"], sorted(pay["sockets"].values()), pay["claimed"],
                       holds, float(base.difficulty.get("span", 80.0)))  # fmt: skip

    # ------------------------------------------------------------------ answers
    def symbolic_answer(self, scene: SceneIR, program: QuestionProgram) -> Answer:
        pay = scene.payload
        fam = program.family
        if fam is Family.P3_ARGMAX_PORT:
            return _label(max(socket_values(pay), key=lambda t: t[1])[0])
        if fam is Family.P4_COUNT_ZONE:
            return _int(red_count(pay))
        g = target_gauge(pay)
        v, zone = float(g["value"]), zone_of(g["value"], tuple(g["zones"]))
        if fam is Family.P1_READ_TRACE:
            return float_answer(v)
        if fam is Family.P2_ZONE_TRACE:
            return zone_answer(zone)
        holds = zone == pay["claimed"]
        return Answer(type=AnswerType.PREMISE, value=PremiseValue(premise_holds=holds, value=v), tolerance=0.0,
                      canonical=f"{'yes' if holds else 'no'}; {format(v, 'g')}")  # fmt: skip

    def geometric_answer(self, scene: SceneIR, program: QuestionProgram) -> Answer | None:
        return None

    # ------------------------------------------------------------------ twins
    def causal_twin(
        self, scene: SceneIR, program: QuestionProgram, seeds: SeedTree
    ) -> tuple[SceneIR, str]:
        pay, rng = copy.deepcopy(scene.payload), seeds.rng()
        tol, fam = float(pay["grade_tol"]), program.family
        v: int | None
        if fam is Family.P3_ARGMAX_PORT:
            sg = strand_gauge(pay)
            order = sorted(sg.values(), key=lambda g: -pay["dials"][g]["value"])
            lead, second = pay["dials"][order[0]], pay["dials"][order[1]]
            margin = float(pay["argmax_margin"])
            up = [v for v in values_for(second, tol) if v >= float(lead["value"]) + margin]
            down = [v for v in values_for(lead, tol) if v <= float(second["value"]) - margin]
            if up:  # runner-up overtakes the leader ...
                g, v = second, up[int(rng.integers(len(up)))]
            elif down:  # ... or the leader drops below the runner-up
                g, v = lead, down[int(rng.integers(len(down)))]
            else:
                raise TwinUnavailable("argmax cannot be flipped")
            g["value"] = float(v)
            if not argmax_ok(pay):
                raise TwinUnavailable("argmax margin")
            return _scene(scene, pay), f"{g['name']} -> {v}"
        if fam is Family.P4_COUNT_ZONE:
            d = pay["dials"][int(rng.integers(pay["k"]))]
            red = zone_of(d["value"], tuple(d["zones"])) == "red"
            v = pick_value(
                rng, d, tol, zone=["green", "amber"][int(rng.integers(2))] if red else "red"
            )
            if v is None:
                raise TwinUnavailable("no zone flip")
            d["value"] = float(v)
            return _scene(scene, pay), f"{d['name']} {'out of' if red else 'into'} red"
        name = target_gauge(pay)["name"]
        d = next(x for x in pay["dials"] if x["name"] == name)
        here = zone_of(d["value"], tuple(d["zones"]))
        if fam is Family.P1_READ_TRACE:
            v = pick_value(rng, d, tol, away_from=float(d["value"]), min_move=5 * tol)
        elif fam is Family.P2_ZONE_TRACE:
            v = pick_value(rng, d, tol, zone=[z for z in ZONES if z != here][int(rng.integers(2))])
        else:
            claimed = pay["claimed"]
            target = (
                [z for z in ZONES if z != claimed][int(rng.integers(2))]
                if here == claimed
                else claimed
            )
            v = pick_value(rng, d, tol, zone=target)
        if v is None:
            raise TwinUnavailable("no admissible needle value")
        old = d["value"]
        d["value"] = float(v)
        return _scene(scene, pay), f"{name} needle {old:g} -> {v}"

    def nuisance_twin(self, scene: SceneIR, program: QuestionProgram, seeds: SeedTree) -> SceneIR:
        pay, rng = copy.deepcopy(scene.payload), seeds.rng()
        pay["style"] = "harness" if pay["style"] == "schematic" else "schematic"
        for d in pay["dials"]:
            d["bezel_rgb"] = list(
                [b for b in BEZELS if list(b) != d["bezel_rgb"]][int(rng.integers(len(BEZELS) - 1))]
            )
        tints = [t for t in TINTS if list(t) != pay["render"]["background_rgb"]]
        pay["render"]["background_rgb"] = list(tints[int(rng.integers(len(tints)))])
        return _scene(scene, pay)


def _label(v: str) -> Answer:
    return Answer(type=AnswerType.LABEL, value=v, canonical=v)


def _int(v: int) -> Answer:
    return Answer(type=AnswerType.INT, value=int(v), canonical=str(int(v)))


def _top_label(pay: dict[str, Any], strand: int) -> str:
    """ONTOP answers name the start terminal's label; unlabelled starts answer with their id (D-14)."""
    return str(pay["sockets"].get(str(strand), f"panel/terminal:start_x{strand}"))


def fault_effective(scene: SceneIR, fault: FaultCode) -> bool:
    return False


PLUGIN = PanelPlugin()

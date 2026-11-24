"""Tiny synthetic bundles drawn with `knotwork.render.canvas` directly (no plugin imports).

Every builder writes a complete bundle directory (beauty, id buffer, masks, manifest, claims) and
returns its path; the expected answers are worked out by hand in the builders' docstrings.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from knotwork.knotcore.contracts import (
    Answer,
    AnswerType,
    Claim,
    ClaimSet,
    GaugeEntry,
    PixelBox,
    TerminalEntry,
    TickEntry,
    ZoneEntry,
    ZoneName,
)
from knotwork.render.canvas import LayeredCanvas

REPO = Path(__file__).resolve().parents[2]
PROFILE: dict[str, Any] = yaml.safe_load(
    (REPO / "configs" / "default_profile.yaml").read_text(encoding="utf-8")
)
CFG: dict[str, Any] = PROFILE["kernel"]
FONT = REPO / PROFILE["render"]["font_path"]
BG = tuple(PROFILE["render"]["background_rgb"])
BLUE, ORANGE, GREEN = (0, 114, 178), (213, 94, 0), (0, 158, 115)


def canvas() -> LayeredCanvas:
    return LayeredCanvas(1024, 2, BG, FONT, PROFILE["render"]["font_sha256"])  # type: ignore[arg-type]


def label_answer(v: str) -> Answer:
    return Answer(type=AnswerType.LABEL, value=v, canonical=v)


def int_answer(v: int) -> Answer:
    return Answer(type=AnswerType.INT, value=v, canonical=str(v))


def write_claims(out: Path, plugin: str, claims: list[Claim]) -> None:
    cs = ClaimSet(plugin=plugin, claims=claims)
    (out / "claims.json").write_text(json.dumps(cs.model_dump(mode="json")), encoding="utf-8")


def polyline(
    x0: float, x1: float, f: Callable[[np.ndarray], np.ndarray], n: int = 241
) -> np.ndarray:
    xs = np.linspace(x0, x1, n)
    return np.stack([xs, f(xs)], axis=1)


@dataclass
class Wire:
    key: str  # start letter
    end: str  # end label
    pts: np.ndarray
    color: tuple[int, int, int] = BLUE
    width: float = 9.0
    beauty_rgb: tuple[int, int, int] | None = None
    gap: tuple[float, float] | None = None  # x-range left undrawn (broken mask)


@dataclass
class Scene:
    wires: list[Wire]
    #: (centre, key of top wire) per crossing; processed in increasing x like the plugins.
    crossings: list[tuple[tuple[float, float], str]] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    extra: Callable[[LayeredCanvas, dict[str, int]], None] | None = None
    end_label_override: dict[str, str] = field(default_factory=dict)
    start_boxes: bool = True


def tid(prefix: str, key: str) -> str:
    return f"fx/terminal:{prefix}_{key}"


def wires_bundle(out: Path, sc: Scene) -> Path:
    """Strands with lettered starts (left end) and numbered ends (right end)."""
    c = canvas()
    ids: dict[str, int] = {}
    for w in sc.wires:
        ids[w.key] = c.new_object(f"fx/strand:{w.key}", "strand", w.color)
    terms: list[TerminalEntry] = []
    for w in sc.wires:
        pieces = [w.pts]
        if w.gap is not None:
            pieces = [w.pts[w.pts[:, 0] <= w.gap[0]], w.pts[w.pts[:, 0] >= w.gap[1]]]
        for piece in pieces:
            c.polyline(ids[w.key], piece, w.width, beauty_rgb=w.beauty_rgb)
    for (cx, cy), top in sorted(sc.crossings, key=lambda t: t[0][0]):
        w = next(x for x in sc.wires if x.key == top)
        c.halo_patch(ids[top], w.pts, w.width, 5, 22, (cx, cy))
    for w in sc.wires:
        (sx, sy), (ex, ey) = w.pts[0], w.pts[-1]
        ls = c.new_object(
            f"fx/label:s_{w.key}", "label", (25, 25, 25), parent="fx/labels:start", label_text=w.key
        )
        c.text(ls, (sx - 38, sy), w.key, 26)
        shown = sc.end_label_override.get(w.end, w.end)
        le = c.new_object(
            f"fx/label:e_{w.key}", "label", (25, 25, 25), parent="fx/labels:end", label_text=w.end
        )
        c.text(le, (ex + 38 if ex > sx else ex - 38, ey), shown, 26)
        box = lambda x, y: PixelBox(x0=int(x) - 10, y0=int(y) - 10, x1=int(x) + 10, y1=int(y) + 10)  # noqa: E731
        terms.append(
            TerminalEntry(terminal_id=tid("s", w.key), role="start", box=box(sx, sy), label_obj=ls)
        )
        terms.append(
            TerminalEntry(terminal_id=tid("e", w.key), role="end", box=box(ex, ey), label_obj=le)
        )
    if sc.extra is not None:
        sc.extra(c, ids)
    c.finish(out, "fx", terminals=terms)
    write_claims(out, "fx", sc.claims)
    return out


def x_scene(top: str = "A", **kw: Any) -> Scene:
    """A: (150,300)->(870,700); B: (150,700)->(870,300). One crossing at (510,500).

    TRACE(A) = "1" (A ends at (870,700) labelled "1"), TRACE(B) = "2", CROSS = 1, ONTOP = top.
    """
    a = polyline(150, 870, lambda x: 300 + (x - 150) * 400 / 720)
    b = polyline(150, 870, lambda x: 700 - (x - 150) * 400 / 720)
    claims = [
        Claim(claim_id="t_a", op="TRACE", args=[tid("s", "A")], expected=label_answer("1")),
        Claim(claim_id="t_b", op="TRACE", args=[tid("s", "B")], expected=label_answer("2")),
        Claim(claim_id="c_a", op="CROSS", args=[tid("s", "A"), "*"], expected=int_answer(1)),
        Claim(claim_id="c_b", op="CROSS", args=[tid("s", "B"), "*"], expected=int_answer(1)),
        Claim(
            claim_id="o_0",
            op="ONTOP",
            args=[tid("s", "A"), tid("s", "B"), 0],
            expected=label_answer(top),
        ),
    ]
    wires = [Wire("A", "1", a, BLUE), Wire("B", "2", b, ORANGE)]
    return Scene(wires=wires, crossings=[((510.0, 500.0), top)], claims=claims, **kw)


def wave_scene(n_cross: int = 3) -> Scene:
    """A is a cosine wave over horizontal B (y=500): crosses at x = 270, 510, 750 (3 crossings).

    n_cross=0 lifts the wave above B (no crossing).
    """
    lift = 0.0 if n_cross else -260.0
    a = polyline(150, 870, lambda x: 500 + lift + 150 * np.cos(3 * math.pi * (x - 150) / 720))
    b = polyline(150, 870, lambda x: 0 * x + 500)
    xs = [270.0, 510.0, 750.0] if n_cross else []
    tops = ["A", "B", "A"]
    crossings = [((x, 500.0), t) for x, t in zip(xs, tops, strict=False)]
    return Scene(wires=[Wire("A", "1", a, BLUE), Wire("B", "2", b, ORANGE)], crossings=crossings)


def doubleback_scene() -> Scene:
    """A runs right along y=300, down x=850, back left along y=650 to (150, 650).

    B is a straight line (700,150)->(300,850): crosses A first at (614,300) then at (414,650).
    Geodesic order along A therefore differs from x order. ONTOP idx 0 -> B, idx 1 -> A.
    """
    seg = lambda p, q, n=120: np.linspace(p, q, n)  # noqa: E731
    a = np.concatenate(
        [
            seg((150, 300), (850, 300)),
            seg((850, 300), (850, 650))[1:],
            seg((850, 650), (150, 650))[1:],
        ]
    )
    b = seg((300, 850), (700, 150), 240)
    claims = [
        Claim(
            claim_id="o0",
            op="ONTOP",
            args=[tid("s", "A"), tid("s", "B"), 0],
            expected=label_answer("B"),
        ),
        Claim(
            claim_id="o1",
            op="ONTOP",
            args=[tid("s", "A"), tid("s", "B"), 1],
            expected=label_answer("A"),
        ),
    ]
    p1 = (700 - 150 * 400 / 700, 300.0)
    p2 = (700 - 500 * 400 / 700, 650.0)
    return Scene(
        wires=[Wire("A", "1", a, BLUE), Wire("B", "2", b, ORANGE)],
        crossings=[(p1, "B"), (p2, "A")],
        claims=claims,
    )


# ----------------------------------------------------------------------------- gauge
@dataclass
class Dial:
    value: float = 47.0
    labels: tuple[str, ...] = ("00", "20", "40", "60", "80")
    vmin: float = 0.0
    vmax: float = 80.0
    zones: tuple[float, float] = (40.0, 60.0)
    pivot_offset: tuple[float, float] = (0.0, 0.0)
    tick_values: tuple[float, ...] | None = None  # override where label i is drawn (scale faults)
    sweep_factor: float = 1.0
    center: tuple[float, float] = (512.0, 512.0)
    radius: float = 340.0
    name: str = "g1"


def angle_of(d: Dial, v: float, sweep_factor: float = 1.0) -> float:
    return 225.0 - (v - d.vmin) / (d.vmax - d.vmin) * 270.0 * sweep_factor


def _pt(c: tuple[float, float], r: float, deg: float) -> tuple[float, float]:
    t = math.radians(deg)
    return c[0] + r * math.cos(t), c[1] - r * math.sin(t)


def draw_dial(c: LayeredCanvas, d: Dial, prefix: str = "fx") -> GaugeEntry:
    cx, cy = d.center
    gid = f"{prefix}/gauge:{d.name}"
    dial = c.new_object(f"{prefix}/dial:{d.name}", "dial", (255, 255, 255), parent=gid)
    c.disk(dial, d.center, d.radius, outline_rgb=(90, 90, 90), outline_px=3)
    r_ = d.radius
    values = [d.vmin + i * (d.vmax - d.vmin) / (len(d.labels) - 1) for i in range(len(d.labels))]
    drawn = list(d.tick_values or values)
    zone_bounds = [d.vmin, d.zones[0], d.zones[1], d.vmax]
    zones = []
    for i, (zn, col) in enumerate(
        zip(("green", "amber", "red"), ((40, 160, 70), (230, 160, 20), (210, 40, 40)), strict=True)
    ):
        z = c.new_object(f"{prefix}/zone_arc:{d.name}_{zn}", "zone_arc", col, parent=gid)
        lo, hi = zone_bounds[i], zone_bounds[i + 1]
        c.arc_band(z, d.center, 0.92 * r_, 1.0 * r_ - 4, angle_of(d, hi), angle_of(d, lo))
        zones.append(ZoneEntry(zone_obj=z, name=ZoneName(zn)))
    ticks = []
    for i, (lab, v) in enumerate(zip(d.labels, drawn, strict=True)):
        deg = angle_of(d, v)
        t = c.new_object(f"{prefix}/tick:{d.name}_{i}", "tick", (40, 40, 40), parent=gid)
        c.polyline(t, np.array([_pt(d.center, 0.78 * r_, deg), _pt(d.center, 0.90 * r_, deg)]), 6)
        lo = c.new_object(
            f"{prefix}/label:{d.name}_t{i}",
            "label",
            (30, 30, 30),
            parent=f"{prefix}/labels:{d.name}",
            label_text=lab,
        )
        c.text(lo, _pt(d.center, 0.62 * r_, deg), lab, 26)
        ticks.append(TickEntry(tick_obj=t, label_obj=lo))
    for i in range(len(values) - 1):
        for k in range(1, 5):
            v = values[i] + k * (values[i + 1] - values[i]) / 5
            deg = angle_of(d, v)
            m = c.new_object(
                f"{prefix}/tick_minor:{d.name}_{i}_{k}", "tick_minor", (90, 90, 90), parent=gid
            )
            c.polyline(
                m, np.array([_pt(d.center, 0.84 * r_, deg), _pt(d.center, 0.90 * r_, deg)]), 3
            )
    deg = angle_of(d, d.value, d.sweep_factor)
    pivot = (cx + d.pivot_offset[0], cy + d.pivot_offset[1])
    nd = c.new_object(f"{prefix}/needle:{d.name}", "needle", (200, 30, 30), parent=gid)
    ux, uy = math.cos(math.radians(deg)), -math.sin(math.radians(deg))
    px, py = -uy, ux
    tail, tip = 0.15 * r_, 0.85 * r_
    poly = [
        (pivot[0] - ux * tail + px * 9, pivot[1] - uy * tail + py * 9),
        (pivot[0] + ux * tip + px * 4, pivot[1] + uy * tip + py * 4),
        (pivot[0] + ux * tip - px * 4, pivot[1] + uy * tip - py * 4),
        (pivot[0] - ux * tail - px * 9, pivot[1] - uy * tail - py * 9),
    ]
    c.polygon(nd, poly)
    hub = c.new_object(f"{prefix}/decor:{d.name}_hub", "decor", (60, 60, 60), parent=gid)
    c.disk(hub, pivot, 12)
    return GaugeEntry(
        gauge_id=gid, dial_obj=dial, needle_obj=nd, ticks=ticks, zones=zones,
        claimed_center=d.center, claimed_radius=d.radius,
    )  # fmt: skip


def zone_of(d: Dial) -> str:
    return "green" if d.value < d.zones[0] else ("amber" if d.value < d.zones[1] else "red")


def gauge_bundle(out: Path, d: Dial | None = None, claims: list[Claim] | None = None) -> Path:
    d = d or Dial()
    c = canvas()
    g = draw_dial(c, d)
    c.finish(out, "fx", gauges=[g])
    if claims is None:
        claims = [
            Claim(claim_id="r", op="READ", args=[g.gauge_id], expected=Answer(type=AnswerType.FLOAT, value=float(d.value), tolerance=0.0, canonical=format(d.value, "g"))),
            Claim(claim_id="z", op="ZONE", args=[g.gauge_id], expected=Answer(type=AnswerType.ZONE, value=zone_of(d), canonical=zone_of(d))),
        ]  # fmt: skip
    write_claims(out, "fx", claims)
    return out


GID = "fx/gauge:g1"


def panel_bundle(out: Path, far: bool = False) -> Path:
    """Two dials with labelled ports below them; sockets S/T cabled to ports G1/G2."""
    c = canvas()
    d1 = Dial(value=30.0, center=(300.0, 330.0), radius=180.0, name="g1")
    d2 = Dial(value=62.0, center=(740.0, 330.0), radius=180.0, name="g2")
    gauges = [draw_dial(c, d1), draw_dial(c, d2)]
    terms = []
    for i, d in enumerate((d1, d2), start=1):
        bottom = int(np.nonzero(c.mask_array(gauges[i - 1].dial_obj).any(axis=1))[0].max())
        y0 = bottom + 6 + (300 if far else 0)
        x = int(d.center[0])
        lab = c.new_object(
            f"fx/label:port{i}", "label", (25, 25, 25), parent="fx/labels:ports", label_text=f"G{i}"
        )
        c.text(lab, (x + 40, y0 + 10), f"G{i}", 26)
        terms.append(
            TerminalEntry(
                terminal_id=f"fx/terminal:p{i}",
                role="port",
                box=PixelBox(x0=x - 10, y0=y0, x1=x + 10, y1=y0 + 20),
                label_obj=lab,
            )
        )
    for i, (key, sy) in enumerate((("S", 800.0), ("T", 900.0)), start=1):
        px = 300.0 if i == 2 else 740.0  # S -> G2, T -> G1 (cables cross)
        py = terms[2 - i].box.y0 + 10
        w = Wire(key, f"G{3 - i}", np.array([[120.0, sy], [px, sy], [px, py]]))
        sid = c.new_object(f"fx/strand:{key}", "strand", (0, 114, 178))
        c.polyline(sid, w.pts, 9)
        lab = c.new_object(
            f"fx/label:s_{key}", "label", (25, 25, 25), parent="fx/labels:start", label_text=key
        )
        c.text(lab, (82, sy), key, 26)
        terms.append(
            TerminalEntry(
                terminal_id=f"fx/terminal:s_{key}",
                role="start",
                box=PixelBox(x0=110, y0=int(sy) - 10, x1=130, y1=int(sy) + 10),
                label_obj=lab,
            )
        )
    c.finish(out, "fx", terminals=terms, gauges=gauges)
    write_claims(out, "fx", [])
    return out

"""`knot` command-line interface (SPEC.md section 9 is the normative contract).

Every subcommand validates its options here, then dispatches to an implementation function
named in DISPATCH_TARGETS. Implementations are built phase by phase (TASKS.md); until a target
module exists the command exits with code 3 and names the task that implements it.

Exit codes: 0 ok | 1 verification/gate failure | 2 usage error | 3 not yet implemented |
            4 safety refusal (budget / checkpoint / API key / network policy)
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import click

from knotwork import __version__

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_IMPLEMENTED, EXIT_SAFETY = 0, 1, 2, 3, 4

#: command -> (module:function, phase, task id in TASKS.md)
DISPATCH_TARGETS: dict[str, tuple[str, str, str]] = {
    "generate": ("knotwork.pipeline.generate:run_generate", "2", "2.9"),
    "verify": ("knotwork.pipeline.verify:run_verify", "3", "3.8"),
    "mutate": ("knotwork.pipeline.mutate:run_mutate", "2", "2.11"),
    "probe": ("knotwork.probes.run:run_probes", "3", "3.4"),
    "diversity": ("knotwork.diversity:run_diversity", "3", "3.5"),
    "export": ("knotwork.export.run:run_export", "4", "4.3"),
    "bench": ("knotwork.bench.run:run_bench", "5", "5.5"),
    "selftest": ("knotwork.pipeline.selftest:run_selftest", "3", "3.7"),
    "report": ("knotwork.report.build:run_report", "6", "6.3"),
    "new-plugin": ("knotwork.scaffold.new_plugin:run_new_plugin", "4", "4.4"),
}

PLUGINS = ("tangle", "gauge", "panel")
FAULTS = tuple(f"M{i}" for i in range(1, 11)) + ("all",)
CHECKPOINT_DIR = Path(".checkpoints")


class PhaseNotImplemented(click.ClickException):
    exit_code = EXIT_NOT_IMPLEMENTED

    def __init__(self, command: str) -> None:
        target, phase, task = DISPATCH_TARGETS[command]
        super().__init__(
            f"E_NOT_IMPLEMENTED: `knot {command}` -> {target} is built in Phase {phase}, "
            f"task {task} (see TASKS.md)."
        )


class SafetyRefusal(click.ClickException):
    exit_code = EXIT_SAFETY


def _dispatch(command: str, **kwargs: Any) -> None:
    target = DISPATCH_TARGETS[command][0]
    mod_name, func_name = target.split(":")
    try:
        module = importlib.import_module(mod_name)
    except ModuleNotFoundError as exc:
        if exc.name and (mod_name == exc.name or mod_name.startswith(exc.name + ".")):
            raise PhaseNotImplemented(command) from None
        raise
    func = getattr(module, func_name, None)
    if func is None:
        raise PhaseNotImplemented(command)
    rc = func(**kwargs)
    sys.exit(int(rc or 0))


def _set_network(policy: str) -> None:
    """Process-wide network policy read by knotwork.net (deny | beacon | openrouter)."""
    os.environ["KNOTWORK_NETWORK"] = policy


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        raise click.UsageError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise click.UsageError(f"config {path} must be a YAML mapping")
    return data


def guard_live(live: bool, budget_usd: float | None, models: Path, dry_run: bool) -> None:
    """Triple lock for paid calls: --live + --budget-usd <= hard cap + human checkpoint file."""
    if not live:
        _set_network("deny")
        return
    if budget_usd is None:
        raise SafetyRefusal("E_BUDGET_REQUIRED: --live requires --budget-usd <cap>.")
    if budget_usd <= 0:
        raise SafetyRefusal("E_BUDGET_INVALID: --budget-usd must be > 0.")
    cap = float(_load_yaml(models).get("budget", {}).get("hard_cap_usd", 0.0))
    if budget_usd > cap:
        raise SafetyRefusal(
            f"E_BUDGET_OVER_CAP: --budget-usd {budget_usd:.2f} exceeds hard_cap_usd {cap:.2f} "
            f"in {models}."
        )
    if dry_run:
        _set_network("deny")
        return
    if not (CHECKPOINT_DIR / "CP2_APPROVED").is_file():
        raise SafetyRefusal(
            "E_CHECKPOINT_REQUIRED: live benchmark needs human approval file "
            ".checkpoints/CP2_APPROVED ([HUMAN-CHECKPOINT-2])."
        )
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SafetyRefusal("E_API_KEY_MISSING: set OPENROUTER_API_KEY for --live.")
    _set_network("openrouter")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="knot")
def cli() -> None:
    """Knotwork proof-carrying visual task factory. Offline by default; zero API keys needed."""
    os.environ.setdefault("KNOTWORK_NETWORK", "deny")


@cli.command()
@click.option("--plugin", type=click.Choice(PLUGINS), required=True)
@click.option("--n", "n", type=click.IntRange(min=1), required=True, help="Certified base items.")
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--seed", type=click.IntRange(min=0), default=None, help="Offline beacon counter.")
@click.option("--beacon", type=click.Choice(["offline", "drand", "nist"]), default="offline")
@click.option(
    "--profile",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("configs/default_profile.yaml"),
)
@click.option("--split", type=click.Choice(["dev", "test", "sealed"]), default="dev")
@click.option("--twins/--no-twins", default=True, help="Emit causal + nuisance twins.")
@click.option("--families", type=str, default=None, help="Comma list; default = profile mix.")
@click.option("--workers", type=click.IntRange(min=0), default=1, help="0 = cpu_count - 1.")
def generate(
    plugin: str,
    n: int,
    out: Path,
    seed: int | None,
    beacon: str,
    profile: Path,
    split: str,
    twins: bool,
    families: str | None,
    workers: int,
) -> None:
    """Generate certified (image, question, answer, evidence) bundles."""
    if beacon != "offline" and seed is not None:
        raise click.UsageError("--seed is only valid with --beacon offline.")
    _set_network("deny" if beacon == "offline" else "beacon")
    fams = [f.strip() for f in families.split(",") if f.strip()] if families else None
    _dispatch(
        "generate",
        plugin=plugin,
        n=n,
        out=out,
        seed=0 if seed is None else seed,
        beacon=beacon,
        profile=profile,
        split=split,
        twins=twins,
        families=fams,
        workers=workers,
    )


@cli.command()
@click.argument("path", type=click.Path(path_type=Path, exists=True))
@click.option("--out", type=click.Path(path_type=Path), default=None)
def verify(path: Path, out: Path | None) -> None:
    """File-only re-verification of a bundle or dataset directory (no plugin code executed)."""
    _set_network("deny")
    _dispatch("verify", path=path, out=out or (path / "verify_report.json"))


@cli.command()
@click.option("--fault", type=click.Choice(FAULTS), default="all")
@click.option("--n", "n", type=click.IntRange(min=1), default=5, help="Injections per fault.")
@click.option("--seed", type=click.IntRange(min=0), default=0)
@click.option(
    "--out", type=click.Path(path_type=Path), default=Path("artifacts/mutate/report.json")
)
@click.option(
    "--profile",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("configs/default_profile.yaml"),
)
def mutate(fault: str, n: int, seed: int, out: Path, profile: Path) -> None:
    """Run the M1-M10 fault-injection conformance matrix."""
    _set_network("deny")
    _dispatch("mutate", fault=fault, n=n, seed=seed, out=out, profile=profile)


@cli.command()
@click.option(
    "--data", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True
)
@click.option(
    "--out", type=click.Path(path_type=Path), default=Path("artifacts/probes/report.json")
)
@click.option("--seed", type=click.IntRange(min=0), default=0)
@click.option("--folds", type=click.IntRange(min=2), default=5)
def probe(data: Path, out: Path, seed: int, folds: int) -> None:
    """Offline shortcut probes (position, color, GBDT image stats, text-only) + balance + dedup."""
    _set_network("deny")
    _dispatch("probe", data=data, out=out, seed=seed, folds=folds)


@cli.command()
@click.option(
    "--data", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True
)
@click.option(
    "--out", type=click.Path(path_type=Path), default=Path("artifacts/diversity/report.json")
)
@click.option("--n", "n", type=click.IntRange(min=16), default=256)
@click.option("--seed", type=click.IntRange(min=0), default=0)
def diversity(data: Path, out: Path, n: int, seed: int) -> None:
    """Vendi Score of the dataset vs. a recolor-only baseline."""
    _set_network("deny")
    _dispatch("diversity", data=data, out=out, n=n, seed=seed)


@cli.command()
@click.option(
    "--data", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True
)
@click.option("--format", "fmt", type=click.Choice(["harbor", "hf", "rlvr", "all"]), default="all")
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--limit", type=click.IntRange(min=1), default=None)
def export(data: Path, fmt: str, out: Path, limit: int | None) -> None:
    """Export certified bundles to Harbor task folders, HF parquet and/or RLVR jsonl."""
    _set_network("deny")
    _dispatch("export", data=data, fmt=fmt, out=out, limit=limit)


@cli.command()
@click.option(
    "--data", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True
)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option(
    "--models",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("configs/models.yaml"),
)
@click.option("--offline/--live", "offline", default=True, help="--live spends real money.")
@click.option("--client", type=click.Choice(["replay", "mock"]), default="replay")
@click.option("--replay-dir", type=click.Path(path_type=Path), default=Path("bench_replay"))
@click.option("--budget-usd", type=float, default=None)
@click.option("--track", type=click.Choice(["tangle", "panel", "all"]), default="all")
@click.option("--max-trials", type=click.IntRange(min=10), default=50)
@click.option("--seed", type=click.IntRange(min=0), default=0)
@click.option("--dry-run", is_flag=True, help="Estimate cost only; never touches the network.")
def bench(
    data: Path,
    out: Path,
    models: Path,
    offline: bool,
    client: str,
    replay_dir: Path,
    budget_usd: float | None,
    track: str,
    max_trials: int,
    seed: int,
    dry_run: bool,
) -> None:
    """Adaptive staircase + 3-VLM diagnostic harness (replay/mock offline; --live gated)."""
    live = not offline
    guard_live(live, budget_usd, models, dry_run)
    _dispatch(
        "bench",
        data=data,
        out=out,
        models=models,
        live=live,
        client="openrouter" if live and not dry_run else client,
        replay_dir=replay_dir,
        budget_usd=budget_usd,
        track=track,
        max_trials=max_trials,
        seed=seed,
        dry_run=dry_run,
    )


@cli.command()
@click.option("--golden", type=click.Path(path_type=Path), default=Path("tests/golden/golden.json"))
@click.option("--update-golden", is_flag=True)
@click.option("--seeds-only", is_flag=True, help="Only the seed/content-address known-answer test.")
def selftest(golden: Path, update_golden: bool, seeds_only: bool) -> None:
    """Golden end-to-end self test (or the seed known-answer vector with --seeds-only)."""
    _set_network("deny")
    if seeds_only:
        from knotwork.knotcore.seeds import SELFTEST_EXPECTED, selftest_vector

        got = selftest_vector()
        ok = bool(SELFTEST_EXPECTED) and got == SELFTEST_EXPECTED
        click.echo(json.dumps({"ok": ok, "vector": got}, sort_keys=True))
        sys.exit(EXIT_OK if ok else EXIT_FAIL)
    _dispatch("selftest", golden=golden, update_golden=update_golden)


@cli.command()
@click.option("--artifacts", type=click.Path(path_type=Path), default=Path("artifacts"))
@click.option(
    "--out", type=click.Path(path_type=Path), default=Path("artifacts/report/report.html")
)
def report(artifacts: Path, out: Path) -> None:
    """Build the self-contained report.html from artifacts/."""
    _set_network("deny")
    _dispatch("report", artifacts=artifacts, out=out)


@cli.command(name="new-plugin")
@click.argument("name")
@click.option("--dest", type=click.Path(path_type=Path), default=Path("src/knotwork/plugins"))
def new_plugin(name: str, dest: Path) -> None:
    """Scaffold a new TaskPlugin package (adds zero lines to knotcore)."""
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,30}", name) or name in PLUGINS:
        raise click.UsageError("NAME must match [a-z][a-z0-9_]{2,30} and not be a core plugin.")
    _dispatch("new-plugin", name=name, dest=dest)


def main() -> None:
    cli(prog_name="knot")


if __name__ == "__main__":
    main()

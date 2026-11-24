"""CLI contract: all 10 subcommands registered, offline defaults, live-mode triple lock."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from knotwork import cli as K

COMMANDS = {
    "generate",
    "verify",
    "mutate",
    "probe",
    "diversity",
    "export",
    "bench",
    "selftest",
    "report",
    "new-plugin",
}


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_all_ten_subcommands_registered(runner: CliRunner) -> None:
    assert set(K.cli.commands) == COMMANDS == set(K.DISPATCH_TARGETS)
    res = runner.invoke(K.cli, ["--help"])
    assert res.exit_code == 0
    for c in COMMANDS:
        assert c in res.output


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_each_subcommand_help(runner: CliRunner, command: str) -> None:
    res = runner.invoke(K.cli, [command, "--help"])
    assert res.exit_code == 0, res.output


def test_dispatch_targets_are_well_formed() -> None:
    for target, phase, task in K.DISPATCH_TARGETS.values():
        mod, func = target.split(":")
        assert mod.startswith("knotwork.") and func.startswith("run_")
        assert phase in {"2", "3", "4", "5", "6"} and task.startswith(f"{phase}.")


def test_seeds_only_selftest_passes(runner: CliRunner) -> None:
    res = runner.invoke(K.cli, ["selftest", "--seeds-only"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["ok"] is True


def test_missing_module_maps_to_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        K.DISPATCH_TARGETS, "report", ("knotwork.does_not_exist.build:run_report", "6", "6.3")
    )
    with pytest.raises(K.PhaseNotImplemented) as exc:
        K._dispatch("report")
    assert exc.value.exit_code == K.EXIT_NOT_IMPLEMENTED
    assert "6.3" in exc.value.message


def test_bench_defaults_to_offline(runner: CliRunner, tmp_path: Path) -> None:
    res = runner.invoke(K.cli, ["bench", "--help"])
    assert "--offline / --live" in res.output


def _bench(runner: CliRunner, tmp_path: Path, repo_root: Path, *extra: str):  # type: ignore[no-untyped-def]
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    args = ["bench", "--data", str(data), "--out", str(tmp_path / "out")]
    args += ["--models", str(repo_root / "configs" / "models.yaml"), *extra]
    return runner.invoke(K.cli, args)


def test_live_requires_budget(runner: CliRunner, tmp_path: Path, repo_root: Path) -> None:
    res = _bench(runner, tmp_path, repo_root, "--live")
    assert res.exit_code == K.EXIT_SAFETY and "E_BUDGET_REQUIRED" in res.output


def test_live_budget_over_cap_refused(runner: CliRunner, tmp_path: Path, repo_root: Path) -> None:
    res = _bench(runner, tmp_path, repo_root, "--live", "--budget-usd", "100")
    assert res.exit_code == K.EXIT_SAFETY and "E_BUDGET_OVER_CAP" in res.output


def test_live_requires_checkpoint_file(
    runner: CliRunner, tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # no .checkpoints/CP2_APPROVED here
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    res = _bench(runner, tmp_path, repo_root, "--live", "--budget-usd", "6")
    assert res.exit_code == K.EXIT_SAFETY and "E_CHECKPOINT_REQUIRED" in res.output


def test_live_requires_api_key(
    runner: CliRunner, tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".checkpoints").mkdir()
    (tmp_path / ".checkpoints" / "CP2_APPROVED").write_text("approved")
    res = _bench(runner, tmp_path, repo_root, "--live", "--budget-usd", "6")
    assert res.exit_code == K.EXIT_SAFETY and "E_API_KEY_MISSING" in res.output


def test_live_dry_run_skips_checkpoint_and_key(
    runner: CliRunner, tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    res = _bench(runner, tmp_path, repo_root, "--live", "--budget-usd", "6", "--dry-run")
    assert res.exit_code in (K.EXIT_OK, K.EXIT_NOT_IMPLEMENTED), res.output
    assert "E_CHECKPOINT_REQUIRED" not in res.output


def test_generate_rejects_seed_with_live_beacon(
    runner: CliRunner, in_repo: Path, tmp_path: Path
) -> None:
    res = runner.invoke(
        K.cli,
        [
            "generate",
            "--plugin",
            "tangle",
            "--n",
            "1",
            "--out",
            str(tmp_path),
            "--beacon",
            "drand",
            "--seed",
            "3",
        ],
    )
    assert res.exit_code == K.EXIT_USAGE


def test_new_plugin_name_validation(runner: CliRunner) -> None:
    assert runner.invoke(K.cli, ["new-plugin", "Bad-Name"]).exit_code == K.EXIT_USAGE
    assert runner.invoke(K.cli, ["new-plugin", "tangle"]).exit_code == K.EXIT_USAGE

"""Task 2.11: `knot mutate` M1-M10 conformance matrix (SPEC 9.1, 10)."""

from __future__ import annotations

from typing import Any

from knotwork.knotcore.contracts import FAULT_EXPECTED, FAULT_PLUGIN


def test_report_schema_and_exact_catch(mutate_run: dict[str, Any]) -> None:
    rep = mutate_run["report"]
    assert mutate_run["rc"] == 0
    assert set(rep) >= {"n_per_fault", "faults", "catch_rate", "exact_catch_rate", "clean"}
    assert rep["n_per_fault"] == 1
    for fault, code in FAULT_EXPECTED.items():
        row = rep["faults"][fault.value]
        assert row["expected"] == code.value and row["plugin"] == FAULT_PLUGIN[fault]
        assert row["injected"] == 1 and row["caught"] == row["caught_exact"] == 1, (fault, row)
        assert set(row) >= {"injected", "caught", "caught_exact", "primary_match", "codes_seen"}
    assert rep["exact_catch_rate"] == 1.0 and rep["catch_rate"] == 1.0
    clean = rep["clean"]
    assert set(clean) >= {"n", "quarantined", "false_quarantine_rate"}
    assert clean["n"] == 10 and clean["false_quarantine_rate"] <= 0.10

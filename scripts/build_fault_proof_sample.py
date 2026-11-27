"""Rebuild the committed default fault proof pair shown when Forensics > Fault proofs opens.

uv run python scripts/build_fault_proof_sample.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from knotwork.studio import fault_proofs


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        sides = fault_proofs.build(fault_proofs.DEFAULT_FAULT, fault_proofs.DEFAULT_SEED, Path(tmp))
    fault_proofs.save_sample(
        fault_proofs.DEFAULT_FAULT, fault_proofs.DEFAULT_SEED, sides, fault_proofs.SAMPLE_DIR
    )
    caught = fault_proofs.caught(fault_proofs.DEFAULT_FAULT, sides[1])
    print(f"wrote {fault_proofs.SAMPLE_DIR}: faulty codes {sides[1].codes}, caught={caught}")
    return 0 if caught and sides[0].verdict == "CERTIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

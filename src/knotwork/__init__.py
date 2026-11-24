"""Knotwork: a proof-carrying synthetic visual task factory.

Package layout (SPEC.md section 2):
    knotwork.knotcore   trusted raster kernel (<= 850 executable / 1350 total LOC; imports no other knotwork pkg)
    knotwork.plugins    untrusted scene plugins (tangle, gauge, panel)
    knotwork.pipeline   generate / certify / mutate / verify / selftest orchestration
    knotwork.freshness  beacons, emission mask, burn ledger, canary
    knotwork.probes     offline shortcut probes
    knotwork.export     Harbor + HuggingFace/RLVR exporters
    knotwork.bench      staircase + VLM harness + statistics
"""

__version__ = "0.1.0"

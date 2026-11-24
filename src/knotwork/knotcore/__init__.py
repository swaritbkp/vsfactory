"""knotcore: the trusted pixel kernel.

Invariants (enforced by tests/conformance and scripts/verify_phase.py):
* <= 850 executable LOC (all modules except declarative contracts.py) and <= 1350 LOC in total
  (non-blank, non-comment, non-docstring lines; see knotwork.devtools.kernel_audit).
* Imports only the standard library (minus networking), numpy, scipy, scikit-image, Pillow,
  pydantic, rapidocr_onnxruntime and knotwork.knotcore.* -- never knotwork.plugins.
* Reads only raster artifacts + manifest/claims/question JSON; never SceneIR.payload.
"""

"""Closed-loop multimodal post-training engine on top of certified Knotwork bundles.

* `vcot`       -- deterministic grounded visual chain-of-thought traces from masks + kernel witness;
* `reward`     -- composite RLVR reward (outcome, spatial grounding, format / abstention), `verl`
                  and TRL compatible, stdlib only so it can be shipped verbatim in the drop-in kit;
* `server`     -- stateless local HTTP verifier (`knot serve-verifier`);
* `curriculum` -- vanishing-advantage frontier analysis, seed-disjoint curriculum and pack export;
* `throughput` -- generation / verification / vCoT throughput and unit economics.

Nothing here modifies `knotcore`; trusted kernel primitives are only imported and called.
"""

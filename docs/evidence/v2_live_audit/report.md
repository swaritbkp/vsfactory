# Model Arena Report — Full v2 4-domain live audit

- **Completed trials:** 530 / 540
- **Certification yield:** 98.3%
- **Best model (95% CI):** gpt-5.4 44.6% [37.5%–52.0%]
- **Total spend:** $0.6727
- **Mean gen / verify:** 1,010 ms / 4,293 ms

## Statistical leaderboard

| Block | Model | Accuracy | 95% CI | Correct/N | p50 latency | Cost | $/correct |
|---|---|---|---|---|---|---|---|
| Rack patch-bay · Clean | openai/gpt-5.4 | 61.1% | 44.9%–75.2% | 22/36 | 3,186 ms | $0.1144 | $0.0052 |
| Rack patch-bay · Clean | google/gemini-2.5-flash | 13.9% | 6.1%–28.7% | 5/36 | 2,791 ms | $0.0173 | $0.0035 |
| Rack patch-bay · Clean | qwen/qwen3-vl-235b-a22b-instruct | 19.4% | 9.8%–35.0% | 7/36 | 6,138 ms | $0.0189 | $0.0027 |
| Rack patch-bay · High stress | openai/gpt-5.4 | 27.3% | 15.1%–44.2% | 9/33 | 5,125 ms | $0.1044 | $0.0116 |
| Rack patch-bay · High stress | google/gemini-2.5-flash | 12.5% | 5.0%–28.1% | 4/32 | 3,220 ms | $0.0151 | $0.0038 |
| Rack patch-bay · High stress | qwen/qwen3-vl-235b-a22b-instruct | 3.0% | 0.5%–15.3% | 1/33 | 9,668 ms | $0.0212 | $0.0212 |
| Tangle · High stress | openai/gpt-5.4 | 47.2% | 32.0%–63.0% | 17/36 | 4,356 ms | $0.1164 | $0.0068 |
| Tangle · High stress | google/gemini-2.5-flash | 33.3% | 20.2%–49.7% | 12/36 | 3,257 ms | $0.0192 | $0.0016 |
| Tangle · High stress | qwen/qwen3-vl-235b-a22b-instruct | 30.6% | 18.0%–46.9% | 11/36 | 7,240 ms | $0.0158 | $0.0014 |
| Gauge · High stress | openai/gpt-5.4 | 75.0% | 58.9%–86.2% | 27/36 | 3,301 ms | $0.0887 | $0.0033 |
| Gauge · High stress | google/gemini-2.5-flash | 63.9% | 47.6%–77.5% | 23/36 | 3,086 ms | $0.0099 | $0.0004 |
| Gauge · High stress | qwen/qwen3-vl-235b-a22b-instruct | 63.9% | 47.6%–77.5% | 23/36 | 3,300 ms | $0.0060 | $0.0003 |
| Panel · Hard logic | openai/gpt-5.4 | 11.1% | 4.4%–25.3% | 4/36 | 3,467 ms | $0.1056 | $0.0264 |
| Panel · Hard logic | google/gemini-2.5-flash | 16.7% | 7.9%–31.9% | 6/36 | 2,980 ms | $0.0117 | $0.0020 |
| Panel · Hard logic | qwen/qwen3-vl-235b-a22b-instruct | 13.9% | 6.1%–28.7% | 5/36 | 4,855 ms | $0.0082 | $0.0016 |

## Reliability over k = 3 repeats (pass@k / pass^k)

| Block | Model | Items | pass@1 | pass@k | pass^k | Gap (pp) |
|---|---|---|---|---|---|---|
| Rack patch-bay · Clean | gpt-5.4 | 12 | 61.1% | 75.0% | 50.0% | 25.0 |
| Rack patch-bay · Clean | gemini-2.5-flash | 12 | 13.9% | 25.0% | 0.0% | 25.0 |
| Rack patch-bay · Clean | qwen3-vl-235b-a22b-instruct | 12 | 19.4% | 25.0% | 16.7% | 8.3 |
| Rack patch-bay · High stress | gpt-5.4 | 11 | 27.3% | 45.5% | 9.1% | 36.4 |
| Rack patch-bay · High stress | gemini-2.5-flash | 10 | 13.3% | 30.0% | 0.0% | 30.0 |
| Rack patch-bay · High stress | qwen3-vl-235b-a22b-instruct | 11 | 3.0% | 9.1% | 0.0% | 9.1 |
| Tangle · High stress | gpt-5.4 | 12 | 47.2% | 83.3% | 16.7% | 66.7 |
| Tangle · High stress | gemini-2.5-flash | 12 | 33.3% | 58.3% | 8.3% | 50.0 |
| Tangle · High stress | qwen3-vl-235b-a22b-instruct | 12 | 30.6% | 50.0% | 8.3% | 41.7 |
| Gauge · High stress | gpt-5.4 | 12 | 75.0% | 91.7% | 58.3% | 33.3 |
| Gauge · High stress | gemini-2.5-flash | 12 | 63.9% | 75.0% | 50.0% | 25.0 |
| Gauge · High stress | qwen3-vl-235b-a22b-instruct | 12 | 63.9% | 75.0% | 58.3% | 16.7 |
| Panel · Hard logic | gpt-5.4 | 12 | 11.1% | 16.7% | 8.3% | 8.3 |
| Panel · Hard logic | gemini-2.5-flash | 12 | 16.7% | 33.3% | 8.3% | 25.0 |
| Panel · Hard logic | qwen3-vl-235b-a22b-instruct | 12 | 13.9% | 16.7% | 8.3% | 8.3 |

> gpt-5.4 on Rack patch-bay · Clean: solves 75.0% of items at least once in 3 tries but only 50.0% on all 3 (gap 25.0 pp). Its answers are not consistent across repeats, so single-run accuracy (61.1%) overstates its reliability.
> gemini-2.5-flash on Rack patch-bay · Clean: solves 25.0% of items at least once in 3 tries but only 0.0% on all 3 (gap 25.0 pp). Its answers are not consistent across repeats, so single-run accuracy (13.9%) overstates its reliability.
> gpt-5.4 on Rack patch-bay · High stress: solves 45.5% of items at least once in 3 tries but only 9.1% on all 3 (gap 36.4 pp). Its answers are not consistent across repeats, so single-run accuracy (27.3%) overstates its reliability.
> gemini-2.5-flash on Rack patch-bay · High stress: solves 30.0% of items at least once in 3 tries but only 0.0% on all 3 (gap 30.0 pp). Its answers are not consistent across repeats, so single-run accuracy (13.3%) overstates its reliability.
> gpt-5.4 on Tangle · High stress: solves 83.3% of items at least once in 3 tries but only 16.7% on all 3 (gap 66.7 pp). Its answers are not consistent across repeats, so single-run accuracy (47.2%) overstates its reliability.
> gemini-2.5-flash on Tangle · High stress: solves 58.3% of items at least once in 3 tries but only 8.3% on all 3 (gap 50.0 pp). Its answers are not consistent across repeats, so single-run accuracy (33.3%) overstates its reliability.
> qwen3-vl-235b-a22b-instruct on Tangle · High stress: solves 50.0% of items at least once in 3 tries but only 8.3% on all 3 (gap 41.7 pp). Its answers are not consistent across repeats, so single-run accuracy (30.6%) overstates its reliability.
> gpt-5.4 on Gauge · High stress: solves 91.7% of items at least once in 3 tries but only 58.3% on all 3 (gap 33.3 pp). Its answers are not consistent across repeats, so single-run accuracy (75.0%) overstates its reliability.
> gemini-2.5-flash on Gauge · High stress: solves 75.0% of items at least once in 3 tries but only 50.0% on all 3 (gap 25.0 pp). Its answers are not consistent across repeats, so single-run accuracy (63.9%) overstates its reliability.
> qwen3-vl-235b-a22b-instruct on Gauge · High stress: solves 75.0% of items at least once in 3 tries but only 58.3% on all 3 (gap 16.7 pp). Its answers are not consistent across repeats, so single-run accuracy (63.9%) overstates its reliability.
> gemini-2.5-flash on Panel · Hard logic: solves 33.3% of items at least once in 3 tries but only 8.3% on all 3 (gap 25.0 pp). Its answers are not consistent across repeats, so single-run accuracy (16.7%) overstates its reliability.

## Cross-block contrasts

| Model | A | B | Effect (pp) | p (z) | p (Fisher) |
|---|---|---|---|---|---|
| openai/gpt-5.4 | Rack patch-bay · Clean | Rack patch-bay · High stress | +33.8 | 0.005 | 0.007 |
| google/gemini-2.5-flash | Rack patch-bay · Clean | Rack patch-bay · High stress | +1.4 | 0.866 | 1.000 |
| qwen/qwen3-vl-235b-a22b-instruct | Rack patch-bay · Clean | Rack patch-bay · High stress | +16.4 | 0.033 | 0.057 |
| openai/gpt-5.4 | Rack patch-bay · Clean | Tangle · High stress | +13.9 | 0.237 | 0.344 |
| google/gemini-2.5-flash | Rack patch-bay · Clean | Tangle · High stress | -19.4 | 0.052 | 0.094 |
| qwen/qwen3-vl-235b-a22b-instruct | Rack patch-bay · Clean | Tangle · High stress | -11.1 | 0.276 | 0.415 |
| openai/gpt-5.4 | Rack patch-bay · Clean | Gauge · High stress | -13.9 | 0.206 | 0.312 |
| google/gemini-2.5-flash | Rack patch-bay · Clean | Gauge · High stress | -50.0 | <0.001 | <0.001 |
| qwen/qwen3-vl-235b-a22b-instruct | Rack patch-bay · Clean | Gauge · High stress | -44.4 | <0.001 | <0.001 |
| openai/gpt-5.4 | Rack patch-bay · Clean | Panel · Hard logic | +50.0 | <0.001 | <0.001 |
| google/gemini-2.5-flash | Rack patch-bay · Clean | Panel · Hard logic | -2.8 | 0.743 | 1.000 |
| qwen/qwen3-vl-235b-a22b-instruct | Rack patch-bay · Clean | Panel · Hard logic | +5.6 | 0.527 | 0.753 |
| openai/gpt-5.4 | Rack patch-bay · High stress | Tangle · High stress | -19.9 | 0.088 | 0.135 |
| google/gemini-2.5-flash | Rack patch-bay · High stress | Tangle · High stress | -20.8 | 0.043 | 0.051 |
| qwen/qwen3-vl-235b-a22b-instruct | Rack patch-bay · High stress | Tangle · High stress | -27.5 | 0.003 | 0.003 |
| openai/gpt-5.4 | Rack patch-bay · High stress | Gauge · High stress | -47.7 | <0.001 | <0.001 |
| google/gemini-2.5-flash | Rack patch-bay · High stress | Gauge · High stress | -51.4 | <0.001 | <0.001 |
| qwen/qwen3-vl-235b-a22b-instruct | Rack patch-bay · High stress | Gauge · High stress | -60.9 | <0.001 | <0.001 |
| openai/gpt-5.4 | Rack patch-bay · High stress | Panel · Hard logic | +16.2 | 0.086 | 0.125 |
| google/gemini-2.5-flash | Rack patch-bay · High stress | Panel · Hard logic | -4.2 | 0.628 | 0.739 |
| qwen/qwen3-vl-235b-a22b-instruct | Rack patch-bay · High stress | Panel · Hard logic | -10.9 | 0.110 | 0.201 |
| openai/gpt-5.4 | Tangle · High stress | Gauge · High stress | -27.8 | 0.016 | 0.029 |
| google/gemini-2.5-flash | Tangle · High stress | Gauge · High stress | -30.6 | 0.009 | 0.018 |
| qwen/qwen3-vl-235b-a22b-instruct | Tangle · High stress | Gauge · High stress | -33.3 | 0.005 | 0.009 |
| openai/gpt-5.4 | Tangle · High stress | Panel · Hard logic | +36.1 | <0.001 | 0.001 |
| google/gemini-2.5-flash | Tangle · High stress | Panel · Hard logic | +16.7 | 0.102 | 0.173 |
| qwen/qwen3-vl-235b-a22b-instruct | Tangle · High stress | Panel · Hard logic | +16.7 | 0.089 | 0.155 |
| openai/gpt-5.4 | Gauge · High stress | Panel · Hard logic | +63.9 | <0.001 | <0.001 |
| google/gemini-2.5-flash | Gauge · High stress | Panel · Hard logic | +47.2 | <0.001 | <0.001 |
| qwen/qwen3-vl-235b-a22b-instruct | Gauge · High stress | Panel · Hard logic | +50.0 | <0.001 | <0.001 |

## Error taxonomy

| Model | wrong_port_or_strand | numeric_tolerance_miss | format_violation | false_premise_error | refusal_or_empty | api_error | Dominant | Hardest bucket |
|---|---|---|---|---|---|---|---|---|
| google/gemini-2.5-flash | 92 | 18 | 6 | 10 | 0 | 1 | wrong_port_or_strand | Panel · Hard logic · P1_READ_TRACE |
| openai/gpt-5.4 | 72 | 14 | 0 | 12 | 0 | 0 | wrong_port_or_strand | Panel · Hard logic · P1_READ_TRACE |
| qwen/qwen3-vl-235b-a22b-instruct | 74 | 19 | 25 | 12 | 0 | 0 | wrong_port_or_strand | Panel · Hard logic · P1_READ_TRACE |

## Factory telemetry

| Block | Items | Certified | Yield | gen ms (mean/p50/p95) | verify ms (mean/p50/p95) |
|---|---|---|---|---|---|
| Rack patch-bay · Clean | 12 | 12 | 100.0% | 832 / 833 / 920 | 2,209 / 2,106 / 4,217 |
| Rack patch-bay · High stress | 12 | 11 | 91.7% | 1,878 / 1,908 / 3,123 | 8,363 / 8,773 / 13,331 |
| Tangle · High stress | 12 | 12 | 100.0% | 735 / 729 / 787 | 2,471 / 2,462 / 2,731 |
| Gauge · High stress | 12 | 12 | 100.0% | 542 / 500 / 743 | 1,444 / 1,309 / 2,020 |
| Panel · Hard logic | 12 | 12 | 100.0% | 1,062 / 993 / 1,429 | 6,976 / 6,435 / 9,538 |
| All | 60 | 59 | 98.3% | 1,010 / 852 / 2,110 | 4,293 / 3,247 / 11,224 |

## Throughput and unit economics

| Measure | Value |
|---|---|
| Items attempted / certified | 60 / 59 |
| Busy time (gen + verify) | 318.1 s |
| Seconds per certified item | 5.392 s |
| Certified items / hour (1 vCPU) | 668 |
| USD per 1M certified samples @ $0.04/vCPU-hr | $59.91 |
| Formula | usd_per_1M_certified = 1e6 * (sum(gen_ms + verify_ms + vcot_ms) / 1000 / certified) / 3600 * vcpu_usd_per_hour |

## Audit header

- **session_name:** `Full v2 4-domain live audit`
- **client:** ``
- **run_id:** `cd038bb66c59`
- **mode:** `live`
- **temperature:** `0.7`
- **started_utc:** `2026-11-27T08:43:59Z`
- **finished_utc:** `2026-11-27T09:01:40Z`
- **git_sha:** `a07908019715cbcc6b02f3119add0c866412b1b1`
- **kernel_sha256:** `cc915c558ec021270fb01b0baab9726af53c21735be08512ed06d36bceadf42f`
- **dataset_sha256:** `05d2391e2577edaf3f524bfd6f389d869743b89f71b8a266a232a2949e30cf6e`
- **data_dir:** `artifacts/arena_workspaces/full_v2_live_eval/data`
- **results_dir:** `artifacts/arena_workspaces/full_v2_live_eval`
- **python:** `3.11.15`
- **platform:** `Windows-10-10.0.26300-SP0`
- **cli:** `uv run python -m knotwork.arena.runner --results-dir "artifacts/arena_workspaces/full_v2_live_eval" --resume`

## Methodology & Literature References

1. Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. JASA 22(158), 209-212. https://doi.org/10.1080/01621459.1927.10502953. Used for every 95% binomial score interval in this report.
2. Roberts, J. et al. ZeroBench: An Impossible Visual Benchmark for Contemporary Large Multimodal Models. ICML 2026, arXiv:2502.09696 (https://arxiv.org/abs/2502.09696). Source of the pass@k / pass^k reliability protocol used when a session asks each item k > 1 times.
3. Rahmanzadehgervi, P. et al. Vision language models are blind. ACCV 2024, arXiv:2407.06581. VLMs fail simple 2-D geometric tasks such as line intersections and path following.
4. Friedman, D. and Dieng, A. B. The Vendi Score: A Diversity Evaluation Metric for Machine Learning. TMLR 2023, arXiv:2210.02410. Knotwork reports Vendi only over kernel-certified items.
5. Lee, K. et al. VLind-Bench: Measuring Language Priors in Large Vision-Language Models. Findings of NAACL 2025, arXiv:2406.08702. Counterfactual tests of language priors.
6. Sun, J. et al. Seeing What Is Actually There: PriVE-Bench and PriVE-Tools for Counterfactual Evaluation of Agentic Visual Evidence in VLMs. EMNLP 2026 (2026.emnlp-main.760).
7. Feng, Y. et al. VisualSphinx: Large-Scale Synthetic Vision Logic Puzzles for RL. arXiv:2505.23977 (2025). Synthetic puzzles filtered with LLM / VLM scoring.

_© 2026 BilotaAI · Team Bilota (Swarit Sharma & Pankaj David) · Knotwork Visual Task Factory_

# Member 2 — CUSUM validation and adversarial robustness

## What is implemented

`run_robustness.py` is a reproducible **controlled detector-level stress-test harness**. It uses the repository's actual `EWMACorrelationBaseline` and `CUSUMChangeDetector` implementations, fixes the EWMA alpha at 0.05 for this test, and generates controlled benign and shifted multivariate feature windows. It covers:

- **Contamination injection:** gradual distribution/correlation shift; compares P1 (EWMA-only adaptation) and P2 (EWMA + CUSUM-gated adaptation).
- **Gradual mimicry:** a slowly increasing signal.
- **CUSUM evasion:** small per-window shifts; also runs a more sensitive CUSUM mitigation (`k=0.25`, `h=2.5`) to measure detection/FPR trade-off.
- **SHAP perturbation proxy:** reorders low-importance entries in a synthetic importance vector and measures Kendall tau. This is explicitly a *proxy*, not a model-generated SHAP experiment.

The script reports detection rate, false-negative rate, false-positive rate on a separate benign replay, time-to-detection, and an operational baseline-absorption time (first sub-threshold window after an alarm). It writes per-window traces, per-seed summaries, aggregated summaries, protocol metadata, and plots to `results/member2/`.

## Run

From the repository root:

```bash
python scripts/member2/run_robustness.py --seeds 1-20
```

Quick smoke test:

```bash
python scripts/member2/run_robustness.py --seeds 1-2 --output-dir results/member2_smoke
```

## Important scientific boundary

The supplied repository has no actual dataset files in `data/ncsrd/` or `data/data4cyber/` (only `.gitkeep` placeholders). Therefore these outputs are synthetic controlled stress-test results and **must not be described as NCSRD-5G/Data4Cyber results or used as final manuscript performance claims**. Dataset-level validation needs the authorized dataset files and the exact stream/window-generation procedure used by the project. The SHAP case also needs a trained model and real SHAP values before it can count as a full SHAP perturbation experiment. The current benign-replay FPR is also synthetic and does not replace dataset-level FPR.

For publication, preserve these files as pilot/implementation-validation evidence, then adapt the same attack schedules to windows from the frozen dataset split and evaluate the three methods with identical windows and attack injection seeds. Report the sensitive-CUSUM mitigation together with its false-alarm cost rather than reporting detection rate alone.

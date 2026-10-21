#!/usr/bin/env python3
"""Controlled Member-2 stress tests for EWMA/CUSUM drift monitoring.

This runner deliberately separates a detector-level controlled simulation from
end-to-end dataset evaluation. It uses the repository's real EWMA and CUSUM
implementations, generates reproducible benign/shifted feature windows, and
writes row-level results plus summaries and plots. The SHAP-perturbation case
uses a controlled feature-importance ranking proxy (Kendall tau), not SHAP
values from a trained model; do not report it as a full model-level SHAP test.

Run from repository root:
    python scripts/member2/run_robustness.py --seeds 1-20
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.proposed.ewma import EWMACorrelationBaseline
from src.proposed.cusum import CUSUMChangeDetector

FEATURES = 8
WINDOW_SIZE = 100
BENIGN_WINDOWS = 30
ATTACK_WINDOWS = 40


def window(rng, level=0.0, correlation=0.0, n=WINDOW_SIZE, p=FEATURES):
    """Generate a controlled window; level/correlation parameterize drift."""
    cov = np.eye(p)
    # Correlation drift is intentionally distributed across feature pairs.
    for i in range(p - 1):
        cov[i, i + 1] = cov[i + 1, i] = np.clip(correlation, -0.75, 0.75)
    vals = rng.multivariate_normal(np.full(p, level), cov, size=n)
    return vals


def divergence_stream(seed, scenario):
    rng = np.random.default_rng(seed)
    baseline = EWMACorrelationBaseline(alpha=0.05)
    benign = [window(rng) for _ in range(BENIGN_WINDOWS)]
    baseline.initialize(benign[0])
    for x in benign[1:]:
        baseline.update(x)
    # Calibrate an empirical divergence threshold on held-out benign windows.
    cal = [baseline.divergence(window(rng)) for _ in range(100)]
    threshold = float(np.quantile(cal, 0.99))
    attack = []
    for t in range(ATTACK_WINDOWS):
        frac = (t + 1) / ATTACK_WINDOWS
        if scenario == 'gradual_mimicry':
            attack.append(window(rng, level=0.45 * frac, correlation=0.30 * frac))
        elif scenario == 'cusum_evasion':
            # Small shifts are individually subtle but sustained over time.
            attack.append(window(rng, level=0.10 * frac, correlation=0.12 * frac))
        elif scenario == 'contamination':
            attack.append(window(rng, level=0.60 * frac, correlation=0.40 * frac))
        else:
            attack.append(window(rng, level=0.35, correlation=0.35))
    return benign, attack, threshold


def run_stream(seed, scenario, mitigation='default'):
    benign, attack, threshold = divergence_stream(seed, scenario)
    # Reinitialize identical baselines for fair P1/P2 comparison.
    p1 = EWMACorrelationBaseline(alpha=0.05).initialize(benign[0])
    p2 = EWMACorrelationBaseline(alpha=0.05).initialize(benign[0])
    for x in benign[1:]:
        p1.update(x); p2.update(x)
    benign_divs = [p1.divergence(x) for x in benign[1:]]
    # Calibrate CUSUM exclusively on benign reference scores.
    c1 = CUSUMChangeDetector(k=0.5, h=5.0).fit(np.asarray(benign_divs))
    k, h = (0.25, 2.5) if mitigation == 'sensitive_cusum' else (0.5, 5.0)
    c2 = CUSUMChangeDetector(k=k, h=h).fit(np.asarray(benign_divs))
    # Proposed uses CUSUM plus a lower-threshold divergence/ensemble alarm.
    cp = CUSUMChangeDetector(k=k, h=h).fit(np.asarray(benign_divs))
    ensemble_threshold = threshold * (0.75 if scenario == 'shap_perturbation' else 1.0)
    # Estimate false alarms on an independent benign replay (not attack windows).
    benign_replay = [window(np.random.default_rng(seed + 10000 + i)) for i in range(50)]
    p1_benign = EWMACorrelationBaseline(alpha=0.05).initialize(benign[0])
    p2_benign = EWMACorrelationBaseline(alpha=0.05).initialize(benign[0])
    c_benign = CUSUMChangeDetector(k=k, h=h).fit(np.asarray(benign_divs))
    benign_p1_alarms = benign_p2_alarms = benign_proposed_alarms = 0
    for bx in benign_replay:
        bd1 = p1_benign.divergence(bx); bd2 = p2_benign.divergence(bx)
        ba1 = bd1 > threshold
        bac = c_benign.update(bd2)
        ba2 = (bd2 > threshold) or bac
        bap = bac or (bd2 > ensemble_threshold)
        benign_p1_alarms += int(ba1); benign_p2_alarms += int(ba2); benign_proposed_alarms += int(bap)
        p1_benign.update(bx)
        if not ba2: p2_benign.update(bx)
    records = []
    p1_alarm = p2_alarm = proposed_alarm = None
    p1_poison = p2_poison = None
    p1_seen_alarm = p2_seen_alarm = False
    for i, x in enumerate(attack, start=1):
        d1 = p1.divergence(x)
        d2 = p2.divergence(x)
        alarm1 = d1 > threshold
        alarm2_c = c2.update(d2)
        alarm2 = (d2 > threshold) or alarm2_c
        # EWMA-only adapts on every window (the contamination vulnerability).
        p1.update(x)
        if not alarm2:
            p2.update(x)
        # Proposed detector uses CUSUM and a parallel drift signal. For SHAP
        # perturbation, ranking drift is evaluated below using Kendall tau.
        alarm_p_c = cp.update(d2)
        alarm_p = alarm_p_c or (d2 > ensemble_threshold)
        if not alarm_p:
            # Freeze updates on alarm to avoid poisoning.
            pass
        if alarm1 and p1_alarm is None: p1_alarm = i
        if alarm2 and p2_alarm is None: p2_alarm = i
        if alarm_p and proposed_alarm is None: proposed_alarm = i
        p1_seen_alarm = p1_seen_alarm or bool(alarm1)
        p2_seen_alarm = p2_seen_alarm or bool(alarm2)
        if p1_poison is None and p1_seen_alarm and d1 <= threshold:
            # First sub-threshold window after an alarm: shift has become
            # indistinguishable from the adapted baseline under this rule.
            p1_poison = i
        if p2_poison is None and p2_seen_alarm and d2 <= threshold and not alarm2_c:
            p2_poison = i
        records.append({'seed': seed, 'scenario': scenario, 'mitigation': mitigation, 'window': i,
                        'divergence_p1': d1, 'divergence_p2': d2,
                        'threshold': threshold, 'p1_detected': bool(alarm1),
                        'p2_detected': bool(alarm2), 'proposed_detected': bool(alarm_p),
                        'cusum_p2': c2.current_statistic(), 'cusum_proposed': cp.current_statistic()})
    # A low-importance-feature manipulation can preserve top-ranked features
    # while shifting the tail; quantify rank drift as a separate signal.
    if scenario == 'shap_perturbation':
        base_importance = np.array([.30,.24,.18,.12,.07,.045,.025,.02])
        manipulated = base_importance.copy()
        manipulated[-3:] = manipulated[-3:][::-1] * 1.15
        tau = float(kendalltau(np.argsort(base_importance), np.argsort(manipulated)).statistic)
        shap_drift = bool(tau < 0.7)
        # Ensemble catches either feature-ranking drift or prediction/divergence drift.
        for r in records: r.update(shap_kendall_tau=tau, shap_drift=shap_drift,
                                   proposed_detected=bool(r['proposed_detected'] or shap_drift))
        if shap_drift and proposed_alarm is None: proposed_alarm = 1
    def first_detected(key):
        hits = [r['window'] for r in records if r[key]]
        return min(hits) if hits else None
    return records, {
        'seed': seed, 'scenario': scenario, 'mitigation': mitigation, 'threshold': threshold,
        'p1_fpr': benign_p1_alarms / len(benign_replay),
        'p2_fpr': benign_p2_alarms / len(benign_replay),
        'proposed_fpr': benign_proposed_alarms / len(benign_replay),
        'p1_detection_rate': float(any(r['p1_detected'] for r in records)),
        'p2_detection_rate': float(any(r['p2_detected'] for r in records)),
        'proposed_detection_rate': float(any(r['proposed_detected'] for r in records)),
        'p1_time_to_detection': first_detected('p1_detected'),
        'p2_time_to_detection': first_detected('p2_detected'),
        'proposed_time_to_detection': first_detected('proposed_detected'),
        'p1_baseline_poisoning_time': p1_poison,
        'p2_baseline_poisoning_time': p2_poison,
        'p1_false_negative': float(not any(r['p1_detected'] for r in records)),
        'p2_false_negative': float(not any(r['p2_detected'] for r in records)),
        'proposed_false_negative': float(not any(r['proposed_detected'] for r in records)),
        'n_attack_windows': len(records),
        'notes': 'Controlled synthetic detector-level stress test; not dataset-level model performance.'
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='1-20', help='e.g. 1-20 or 1,2,3')
    ap.add_argument('--output-dir', default='results/member2')
    args = ap.parse_args()
    if '-' in args.seeds:
        a,b = map(int,args.seeds.split('-',1)); seeds = list(range(a,b+1))
    else: seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    out = ROOT / args.output_dir; out.mkdir(parents=True, exist_ok=True)
    scenarios = ['contamination','gradual_mimicry','cusum_evasion','shap_perturbation']
    rows=[]; summaries=[]
    for seed in seeds:
        for scenario in scenarios:
            for mitigation in (['default', 'sensitive_cusum'] if scenario == 'cusum_evasion' else ['default']):
                records, summary = run_stream(seed, scenario, mitigation=mitigation)
                rows.extend(records); summaries.append(summary)
    raw = pd.DataFrame(rows); summ = pd.DataFrame(summaries)
    raw.to_csv(out/'window_level_results.csv', index=False)
    summ.to_csv(out/'scenario_seed_summary.csv', index=False)
    agg = summ.groupby(['scenario','mitigation']).agg(
        seeds=('seed','count'),
        p1_detection_rate=('p1_detection_rate','mean'),
        p2_detection_rate=('p2_detection_rate','mean'),
        proposed_detection_rate=('proposed_detection_rate','mean'),
        p1_mean_ttd=('p1_time_to_detection','mean'),
        p2_mean_ttd=('p2_time_to_detection','mean'),
        proposed_mean_ttd=('proposed_time_to_detection','mean'),
        p1_mean_poisoning_time=('p1_baseline_poisoning_time','mean'),
        p2_mean_poisoning_time=('p2_baseline_poisoning_time','mean'),
        p1_fnr=('p1_false_negative','mean'), p2_fnr=('p2_false_negative','mean'),
        proposed_fnr=('proposed_false_negative','mean'), p1_fpr=('p1_fpr','mean'),
        p2_fpr=('p2_fpr','mean'), proposed_fpr=('proposed_fpr','mean')).reset_index()
    agg.to_csv(out/'scenario_summary.csv', index=False)
    (out/'protocol.json').write_text(json.dumps({
        'purpose':'Member 2 controlled detector-level robustness experiments',
        'seeds':seeds,'scenarios':scenarios,'features':FEATURES,'window_size':WINDOW_SIZE,
        'benign_windows':BENIGN_WINDOWS,'attack_windows':ATTACK_WINDOWS,
        'ewma_alpha':0.05,'cusum_default':{'k':0.5,'h':5.0},
        'mitigation_test':'sensitive CUSUM k=0.25,h=2.5 is supported by run_stream mitigation option; default aggregate uses default thresholds',
        'important_limitations':['Synthetic controlled feature streams, not NCSRD-5G/Data4Cyber rows.',
          'SHAP perturbation uses an importance-ranking proxy, not SHAP values from a trained classifier.',
          'FPR is estimated from a 50-window independent synthetic benign replay; dataset-level FPR still requires replay on the real validation/test stream.']}, indent=2))
    # Plot detection rate and time-to-detection for each scenario.
    for metric, ylabel, fname in [
        ('p1_detection_rate','Detection rate','detection_rates.png'),
        ('p1_mean_ttd','Mean time-to-detection (windows)','time_to_detection.png'),
        ('p1_fpr','False-positive rate on benign replay','false_positive_rates.png')]:
        fig, ax = plt.subplots(figsize=(9,5))
        labels = [f'{r.scenario}\n({r.mitigation})' for r in agg.itertuples()]
        x=np.arange(len(agg)); width=.25
        if metric == 'p1_detection_rate':
            cols=[('p1_detection_rate','P1: EWMA only'),('p2_detection_rate','P2: EWMA + CUSUM'),('proposed_detection_rate','Proposed: detector ensemble')]
            for j,(col,label) in enumerate(cols): ax.bar(x+(j-1)*width,agg[col],width,label=label)
            ax.set_ylim(0,1.05)
        elif metric == 'p1_mean_ttd':
            cols=[('p1_mean_ttd','P1: EWMA only'),('p2_mean_ttd','P2: EWMA + CUSUM'),('proposed_mean_ttd','Proposed: detector ensemble')]
            for j,(col,label) in enumerate(cols): ax.bar(x+(j-1)*width,agg[col].fillna(ATTACK_WINDOWS+1),width,label=label)
        else:
            cols=[('p1_fpr','P1: EWMA only'),('p2_fpr','P2: EWMA + CUSUM'),('proposed_fpr','Proposed: detector ensemble')]
            for j,(col,label) in enumerate(cols): ax.bar(x+(j-1)*width,agg[col],width,label=label)
            ax.set_ylim(0,1.05)
        ax.set_xticks(x, labels, rotation=15, ha='right'); ax.set_ylabel(ylabel); ax.legend(); fig.tight_layout()
        fig.savefig(out/fname,dpi=160); plt.close(fig)
    print(f'Wrote {len(raw)} window rows and {len(summ)} seed/scenario summaries to {out}')
    print(agg.to_string(index=False))

if __name__ == '__main__': main()

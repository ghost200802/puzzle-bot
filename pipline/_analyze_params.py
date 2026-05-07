import json
import numpy as np
import sys

REPORT_PATH = '../output/puzzle_new/5_connectivity/texture_verify_report.json'
with open(REPORT_PATH) as f:
    data = json.load(f)

records = []
for pid, sides in data.items():
    for si, matches in enumerate(sides):
        for m in matches:
            if m['reason'] in ('ok', 'color_reject', 'color_gradient_reject'):
                records.append(m)

print(f"Total verified records: {len(records)}")

best_matches = {}
for pid, sides in data.items():
    for si, matches in enumerate(sides):
        valid = [m for m in matches if m['reason'] in ('ok','color_reject','color_gradient_reject')]
        if valid:
            best = min(valid, key=lambda x: x['error'])
            best_matches[(pid, si)] = best

print(f"Best matches (unique piece-sides): {len(best_matches)}")

best_recs = list(best_matches.values())
best_pids = set((pid, si) for pid, si in best_matches.keys())

non_best = []
for r in records:
    non_best.append(r)

print("\n" + "="*70)
print("COMPARISON: Best match (likely true) vs All matches")
print("="*70)

for label, recs in [("Best (n={})".format(len(best_recs)), best_recs), 
                     ("All  (n={})".format(len(records)), records)]:
    cd = np.array([m['color_diff'] for m in recs])
    gs_vals = [m['grad_score'] for m in recs if m['grad_score'] is not None]
    gs = np.array(gs_vals)
    err = np.array([m['error'] for m in recs])
    
    print(f"\n--- {label} ---")
    print(f"  color_diff: mean={np.mean(cd):.1f} P5={np.percentile(cd,5):.1f} P10={np.percentile(cd,10):.1f} P25={np.percentile(cd,25):.1f} P50={np.percentile(cd,50):.1f} P75={np.percentile(cd,75):.1f} P90={np.percentile(cd,90):.1f} P95={np.percentile(cd,95):.1f}")
    print(f"  grad_score: mean={np.mean(gs):.3f} P5={np.percentile(gs,5):.3f} P10={np.percentile(gs,10):.3f} P25={np.percentile(gs,25):.3f} P50={np.percentile(gs,50):.3f} P75={np.percentile(gs,75):.3f} P90={np.percentile(gs,90):.3f} P95={np.percentile(gs,95):.3f}")
    print(f"  error:      mean={np.mean(err):.0f} P5={np.percentile(err,5):.0f} P50={np.percentile(err,50):.0f} P95={np.percentile(err,95):.0f}")

print("\n" + "="*70)
print("color_diff histogram (Best vs All)")
print("="*70)
bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250]
for label, recs in [("Best", best_recs), ("All ", records)]:
    cd = np.array([m['color_diff'] for m in recs])
    hist, _ = np.histogram(cd, bins=bins)
    total = len(cd)
    line = f"{label}: "
    for i in range(len(hist)):
        pct = hist[i] / total * 100
        bar = "#" * int(pct)
        line += f"\n  [{bins[i]:3d}-{bins[i+1]:3d}): {hist[i]:5d} ({pct:5.1f}%) {bar}"
    print(line)

print("\n" + "="*70)
print("grad_score histogram (Best vs All, rich texture only)")
print("="*70)
gs_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for label, recs in [("Best", best_recs), ("All ", records)]:
    gs_vals = [m['grad_score'] for m in recs if m['grad_score'] is not None]
    gs = np.array(gs_vals)
    hist, _ = np.histogram(gs, bins=gs_bins)
    total = len(gs)
    line = f"{label} (n={total}): "
    for i in range(len(hist)):
        pct = hist[i] / total * 100
        bar = "#" * int(pct)
        line += f"\n  [{gs_bins[i]:.1f}-{gs_bins[i+1]:.1f}): {hist[i]:5d} ({pct:5.1f}%) {bar}"
    print(line)

print("\n" + "="*70)
print("2D distribution: color_diff vs error (for best matches)")
print("="*70)
for m in sorted(best_recs, key=lambda x: x['color_diff'])[:20]:
    print(f"  cd={m['color_diff']:6.1f}  gs={m['grad_score'] if m['grad_score'] is not None else 'N/A':>6}  err={m['error']:5.0f}  tex={m['texture_level']}")

print("\n... and bottom 20 (highest color_diff):")
for m in sorted(best_recs, key=lambda x: x['color_diff'])[-20:]:
    print(f"  cd={m['color_diff']:6.1f}  gs={m['grad_score'] if m['grad_score'] is not None else 'N/A':>6}  err={m['error']:5.0f}  tex={m['texture_level']}")

print("\n" + "="*70)
print("color_diff-only rejection simulation (NO grad_score gate)")
print("="*70)
for cd_thresh in [30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100]:
    rej_all = sum(1 for m in records if m['color_diff'] > cd_thresh)
    rej_best = sum(1 for m in best_recs if m['color_diff'] > cd_thresh)
    print(f"  cd>{cd_thresh:3d}: reject {rej_all:5d}/{len(records)} ({rej_all/len(records)*100:.1f}%)  "
          f"best_rejected={rej_best}/{len(best_recs)} ({rej_best/len(best_recs)*100:.1f}%)")

print("\n" + "="*70)
print("Combined: color_diff OR grad_score rejection")
print("="*70)
for cd_thresh in [40, 50, 60, 70, 80]:
    for gs_thresh in [0.3, 0.35, 0.4, 0.45]:
        rej_all = 0
        rej_best = 0
        for m in records:
            reject = False
            if m['color_diff'] > cd_thresh:
                reject = True
            if m['grad_score'] is not None and m['grad_score'] < gs_thresh:
                reject = True
            if reject:
                rej_all += 1
        for m in best_recs:
            reject = False
            if m['color_diff'] > cd_thresh:
                reject = True
            if m['grad_score'] is not None and m['grad_score'] < gs_thresh:
                reject = True
            if reject:
                rej_best += 1
        print(f"  cd>{cd_thresh} OR gs<{gs_thresh}: reject {rej_all:5d}/{len(records)} ({rej_all/len(records)*100:.1f}%)  "
              f"best_rej={rej_best}/{len(best_recs)} ({rej_best/len(best_recs)*100:.1f}%)")

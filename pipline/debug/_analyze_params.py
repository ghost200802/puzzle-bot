import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..', 'src'))
import json
import numpy as np

from pipline.config import get_connectivity_path

REPORT_PATH = os.path.join(get_connectivity_path(), 'texture_verify_report.json')
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

best_recs = list(best_matches.values())
print(f"Best matches (likely true): {len(best_recs)}")

print("\n" + "="*70)
print("NCC distribution: Best match (true) vs All matches")
print("="*70)

for label, recs in [("Best (n={})".format(len(best_recs)), best_recs),
                     ("All  (n={})".format(len(records)), records)]:
    ncc = np.array([m['ncc'] for m in recs])
    cd = np.array([m['color_diff'] for m in recs])
    print(f"\n--- {label} ---")
    print(f"  NCC:        mean={np.mean(ncc):.3f} P5={np.percentile(ncc,5):.3f} P10={np.percentile(ncc,10):.3f} P25={np.percentile(ncc,25):.3f} P50={np.percentile(ncc,50):.3f} P75={np.percentile(ncc,75):.3f} P90={np.percentile(ncc,90):.3f} P95={np.percentile(ncc,95):.3f}")
    print(f"  color_diff: mean={np.mean(cd):.1f} P50={np.percentile(cd,50):.1f}")

print("\n" + "="*70)
print("NCC histogram (Best vs All)")
print("="*70)
ncc_bins = [-1.0, -0.5, -0.2, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for label, recs in [("Best", best_recs), ("All ", records)]:
    ncc = np.array([m['ncc'] for m in recs])
    hist, _ = np.histogram(ncc, bins=ncc_bins)
    total = len(ncc)
    line = f"{label} (n={total}):"
    for i in range(len(hist)):
        pct = hist[i] / total * 100
        bar = "#" * int(pct)
        line += f"\n  [{ncc_bins[i]:5.1f},{ncc_bins[i+1]:5.1f}): {hist[i]:5d} ({pct:5.1f}%) {bar}"
    print(line)

print("\n" + "="*70)
print("Best matches sorted by NCC (bottom 20 = lowest NCC)")
print("="*70)
for m in sorted(best_recs, key=lambda x: x['ncc'])[:20]:
    gs_str = f"{m['grad_score']:.3f}" if m['grad_score'] is not None else "N/A"
    print(f"  ncc={m['ncc']:6.3f}  cd={m['color_diff']:6.1f}  gs={gs_str:>6}  err={m['error']:5.0f}  tex={m['texture_level']}")

print("\n... and top 20 (highest NCC):")
for m in sorted(best_recs, key=lambda x: x['ncc'])[-20:]:
    gs_str = f"{m['grad_score']:.3f}" if m['grad_score'] is not None else "N/A"
    print(f"  ncc={m['ncc']:6.3f}  cd={m['color_diff']:6.1f}  gs={gs_str:>6}  err={m['error']:5.0f}  tex={m['texture_level']}")

print("\n" + "="*70)
print("NCC-only rejection simulation")
print("="*70)
for ncc_thresh in [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    rej_all = sum(1 for m in records if m['ncc'] < ncc_thresh)
    rej_best = sum(1 for m in best_recs if m['ncc'] < ncc_thresh)
    print(f"  ncc<{ncc_thresh:.2f}: reject {rej_all:5d}/{len(records)} ({rej_all/len(records)*100:.1f}%)  "
          f"best_rej={rej_best}/{len(best_recs)} ({rej_best/len(best_recs)*100:.1f}%)")

print("\n" + "="*70)
print("Combined: NCC AND color_diff rejection")
print("="*70)
for ncc_thresh in [0.2, 0.25, 0.3, 0.35, 0.4]:
    for cd_thresh in [50, 60, 70, 80, 90]:
        rej_all = sum(1 for m in records if m['ncc'] < ncc_thresh and m['color_diff'] > cd_thresh)
        rej_best = sum(1 for m in best_recs if m['ncc'] < ncc_thresh and m['color_diff'] > cd_thresh)
        print(f"  ncc<{ncc_thresh} AND cd>{cd_thresh}: reject {rej_all:5d}/{len(records)} ({rej_all/len(records)*100:.1f}%)  "
              f"best_rej={rej_best}/{len(best_recs)} ({rej_best/len(best_recs)*100:.1f}%)")

print("\n" + "="*70)
print("Combined: NCC OR color_diff rejection")
print("="*70)
for ncc_thresh in [0.1, 0.15, 0.2, 0.25, 0.3]:
    for cd_thresh in [80, 100, 120, 140]:
        rej_all = sum(1 for m in records if m['ncc'] < ncc_thresh or m['color_diff'] > cd_thresh)
        rej_best = sum(1 for m in best_recs if m['ncc'] < ncc_thresh or m['color_diff'] > cd_thresh)
        print(f"  ncc<{ncc_thresh} OR cd>{cd_thresh}: reject {rej_all:5d}/{len(records)} ({rej_all/len(records)*100:.1f}%)  "
              f"best_rej={rej_best}/{len(best_recs)} ({rej_best/len(best_recs)*100:.1f}%)")

print("\n" + "="*70)
print("Cross-tab: error range vs NCC")
print("="*70)
for err_thresh in [500, 800, 1000, 1200, 1500]:
    subset = [m for m in records if m['error'] <= err_thresh]
    if not subset:
        continue
    nccs = np.array([m['ncc'] for m in subset])
    print(f"  error<={err_thresh} (n={len(subset)}): NCC mean={np.mean(nccs):.3f} P50={np.median(nccs):.3f} P10={np.percentile(nccs,10):.3f} P90={np.percentile(nccs,90):.3f}")

import json
import numpy as np

with open('output/puzzle_new/dedup_match_meta.json') as f:
    meta = json.load(f)

confirmed_ncc = []
rejected_ncc = []
for key, m in meta.items():
    ncc = m['ncc']
    if m['confirmed']:
        confirmed_ncc.append(ncc)
    else:
        rejected_ncc.append(ncc)

confirmed_ncc = np.array(confirmed_ncc)
rejected_ncc = np.array(rejected_ncc)

print('=' * 60)
print('NCC Distribution Analysis')
print('=' * 60)

print(f'\nConfirmed pairs: {len(confirmed_ncc)}')
print(f'  min={confirmed_ncc.min():.4f}, max={confirmed_ncc.max():.4f}, mean={confirmed_ncc.mean():.4f}, median={np.median(confirmed_ncc):.4f}')
print(f'  p10={np.percentile(confirmed_ncc,10):.4f}, p25={np.percentile(confirmed_ncc,25):.4f}, p50={np.percentile(confirmed_ncc,50):.4f}, p75={np.percentile(confirmed_ncc,75):.4f}, p90={np.percentile(confirmed_ncc,90):.4f}')

print(f'\nRejected pairs: {len(rejected_ncc)}')
print(f'  min={rejected_ncc.min():.4f}, max={rejected_ncc.max():.4f}, mean={rejected_ncc.mean():.4f}, median={np.median(rejected_ncc):.4f}')
print(f'  p10={np.percentile(rejected_ncc,10):.4f}, p25={np.percentile(rejected_ncc,25):.4f}, p50={np.percentile(rejected_ncc,50):.4f}, p75={np.percentile(rejected_ncc,75):.4f}, p90={np.percentile(rejected_ncc,90):.4f}')

print(f'\n--- NCC Histogram (bins of 0.05) ---')
bins = np.arange(0, 1.01, 0.05)
header = f'{"range":>18s} | conf | rejct | total'
print(header)
print('-' * 55)
for i in range(len(bins) - 1):
    lo, hi = bins[i], bins[i + 1]
    c = int(np.sum((confirmed_ncc >= lo) & (confirmed_ncc < hi)))
    r = int(np.sum((rejected_ncc >= lo) & (rejected_ncc < hi)))
    t = c + r
    if t > 0:
        bar_c = '#' * c
        bar_r = '.' * min(r, 60)
        print(f'  [{lo:.2f}, {hi:.2f}) | {c:4d} | {r:5d} | {t:5d}  {bar_c}{bar_r}')

print(f'\n--- Natural Gap Analysis ---')
all_vals = np.sort(np.concatenate([confirmed_ncc, rejected_ncc]))
gaps = np.diff(all_vals)
top_gaps = np.argsort(gaps)[-5:][::-1]
for rank, gi in enumerate(top_gaps):
    print(f'  Gap #{rank+1}: NCC jumps from {all_vals[gi]:.4f} to {all_vals[gi+1]:.4f} (gap={gaps[gi]:.4f})')
    c_below = int(np.sum(confirmed_ncc <= all_vals[gi]))
    c_above = int(np.sum(confirmed_ncc > all_vals[gi]))
    r_below = int(np.sum(rejected_ncc <= all_vals[gi]))
    r_above = int(np.sum(rejected_ncc > all_vals[gi]))
    print(f'    confirmed: {c_below} below, {c_above} above')
    print(f'    rejected:  {r_below} below, {r_above} above')

print(f'\n--- Best Threshold Search ---')
results = []
for thresh in np.arange(0.30, 0.96, 0.005):
    c_below = int(np.sum(confirmed_ncc < thresh))
    r_above = int(np.sum(rejected_ncc >= thresh))
    mis = c_below + r_above
    c_above = int(np.sum(confirmed_ncc >= thresh))
    r_below = int(np.sum(rejected_ncc < thresh))
    results.append((mis, thresh, c_below, r_above, c_above, r_below))

results.sort(key=lambda x: (x[0], -x[1]))
print(f'  {"thresh":>6s} | misclass | conf<Th | rej>=Th | conf>=Th | rej<Th')
print(f'  {"":>6s} |          | (error) | (error) | (ok)    | (ok)')
print('  ' + '-' * 65)
for mis, thresh, cb, ra, ca, rb in results[:15]:
    print(f'  {thresh:.3f}  |   {mis:3d}    |   {cb:3d}  |   {ra:3d}  |   {ca:3d}   |  {rb:3d}')

print(f'\n--- 3-Class Clustering ---')
sorted_all = np.sort(all_vals)
n = len(sorted_all)
best_score = -1
best_t1 = 0
best_t2 = 0
for t1_idx in range(n // 4, 3 * n // 4):
    for t2_idx in range(t1_idx + 1, min(t1_idx + n // 2, n - 1)):
        t1 = sorted_all[t1_idx]
        t2 = sorted_all[t2_idx]
        low_c = int(np.sum(confirmed_ncc < t1))
        mid_c = int(np.sum((confirmed_ncc >= t1) & (confirmed_ncc < t2)))
        high_c = int(np.sum(confirmed_ncc >= t2))
        low_r = int(np.sum(rejected_ncc < t1))
        mid_r = int(np.sum((rejected_ncc >= t1) & (rejected_ncc < t2)))
        high_r = int(np.sum(rejected_ncc >= t2))
        purity = (high_c + low_r) / max(len(confirmed_ncc) + len(rejected_ncc), 1)
        if purity > best_score:
            best_score = purity
            best_t1 = t1
            best_t2 = t2
            best_stats = (low_c, mid_c, high_c, low_r, mid_r, high_r)

lc, mc, hc, lr, mr, hr = best_stats
print(f'  Best 3-class split: [{0:.2f}, {best_t1:.4f}) | [{best_t1:.4f}, {best_t2:.4f}) | [{best_t2:.4f}, 1.0)')
print(f'  Confirmed: low={lc}, mid={mc}, high={hc}')
print(f'  Rejected:  low={lr}, mid={mr}, high={hr}')
print(f'  Purity: {best_score:.4f}')
print(f'  Interpretation:')
print(f'    LOW  (NCC < {best_t1:.3f}): clearly different ({lr} rejected, {lc} confirmed)')
print(f'    MID  ({best_t1:.3f} <= NCC < {best_t2:.3f}): ambiguous ({mr} rejected, {mc} confirmed)')
print(f'    HIGH (NCC >= {best_t2:.3f}): clearly same ({hc} confirmed, {hr} rejected)')

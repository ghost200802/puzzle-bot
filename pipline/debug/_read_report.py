import json
d = json.load(open('output/puzzle_new/6_solution/target_match/target_match_report.json'))
scores = [v['score'] for v in d.values()]
print(f'count={len(scores)}, min={min(scores):.3f}, max={max(scores):.3f}, mean={sum(scores)/len(scores):.3f}')
has_dx = 'dx' in list(d.values())[0]
if has_dx:
    dx_vals = [v['dx'] for v in d.values()]
    dy_vals = [v['dy'] for v in d.values()]
    ang_vals = [v['angle'] for v in d.values()]
    print(f'dx: min={min(dx_vals):.1f}, max={max(dx_vals):.1f}, mean={sum(dx_vals)/len(dx_vals):.1f}')
    print(f'dy: min={min(dy_vals):.1f}, max={max(dy_vals):.1f}, mean={sum(dy_vals)/len(dy_vals):.1f}')
    print(f'angle: min={min(ang_vals):.1f}, max={max(ang_vals):.1f}, mean={sum(ang_vals)/len(ang_vals):.1f}')
high = sum(1 for s in scores if s >= 0.5)
print(f'score>=0.5: {high}/{len(scores)}')
top = sorted(d.items(), key=lambda x: x[1]['score'], reverse=True)[:5]
for pid, r in top:
    print(f'  piece {pid}: score={r["score"]:.3f} dx={r.get("dx",0):.1f} dy={r.get("dy",0):.1f} angle={r.get("angle",0):.1f}')

import json

with open('../output/puzzle_new/5_connectivity/connectivity.json') as f:
    d = json.load(f)

for pid_str in ['137', '141']:
    pid = int(pid_str)
    info = d[pid_str]
    print(f"Piece {pid}:")
    for i, s in enumerate(info):
        if s is None:
            print(f"  Side {i}: EDGE")
        else:
            top = [(m['pid'], m['si'], round(m['error'], 3)) for m in s[:5]]
            print(f"  Side {i}: {len(s)} matches, top5: {top}")
    print()

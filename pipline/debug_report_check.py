import os
import json

_here = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_here, '..', 'output', 'puzzle_new')
CONNECTIVITY_PATH = os.path.join(OUTPUT_DIR, '5_connectivity')

report_path = os.path.join(CONNECTIVITY_PATH, 'texture_verify_report.json')
with open(report_path) as f:
    report = json.load(f)

# Check piece 4 and 137
for pid in [4, 137]:
    pid_s = str(pid)
    if pid_s not in report:
        print(f"Piece {pid}: NOT FOUND")
        continue
    print(f"\nPiece {pid}:")
    for si in range(4):
        matches = report[pid_s][si]
        if not matches:
            print(f"  Side {si}: no matches")
            continue
        for m in matches:
            other_pid = m['pid']
            other_si = m['si']
            is_pair = (pid == 4 and other_pid == 137) or (pid == 137 and other_pid == 4)
            marker = " <<<" if is_pair else ""
            print(f"  Side {si} -> pid={other_pid}[{other_si}]  "
                  f"error={m['error']}  ncc={m['ncc']}  "
                  f"color_diff={m['color_diff']}  grad={m['grad_score']}  "
                  f"reject={m['reject']}  reason={m['reason']}{marker}")

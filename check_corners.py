import os, json

d = 'output/puzzle_run/3_vector'

# These were the "corners" from no-match analysis
test_pieces = [13, 43, 46, 84, 86]

for pid in test_pieces:
    edge_flags = []
    for si in range(4):
        fp = os.path.join(d, f'side_{pid}_{si}.json')
        with open(fp) as f:
            data = json.load(f)
        edge_flags.append(data['is_edge'])
    print(f"Piece {pid}: is_edge={edge_flags}, sum={sum(edge_flags)}")

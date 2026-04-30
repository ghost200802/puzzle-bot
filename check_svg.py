import json, os

d = 'output/puzzle_run/3_vector'
for i in range(4):
    with open(os.path.join(d, f'side_1_{i}.json')) as f:
        data = json.load(f)
    print(f"side_{i}: is_edge={data['is_edge']}, vertices={len(data['vertices'])}, "
          f"piece_center={data.get('piece_center')}, "
          f"first3={data['vertices'][:3]}, last3={data['vertices'][-3:]}")

print("\n--- For comparison, piece 2 ---")
for i in range(4):
    with open(os.path.join(d, f'side_2_{i}.json')) as f:
        data = json.load(f)
    print(f"side_{i}: is_edge={data['is_edge']}, vertices={len(data['vertices'])}")

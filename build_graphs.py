

import pickle
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True,
                    help='Tmall or lastfm')
parser.add_argument('--sample_num', type=int, default=12,
                    help='Number of neighbors to keep per item (default: 12)')
parser.add_argument('--window', type=int, default=3,
                    help='Co-occurrence window size (default: 3)')
opt = parser.parse_args()

# Dataset-specific num_node (must match main.py)
NUM_NODES = {
    'tmall':       40728,
    'lastfm':      38616,
    'retailrocket': 36969,
}

data_dir   = 'datasets/' + opt.dataset
seq_file   = data_dir + '/all_train_seq.txt'
adj_file   = data_dir + '/adj_' + str(opt.sample_num) + '.pkl'
num_file   = data_dir + '/num_' + str(opt.sample_num) + '.pkl'

key = opt.dataset.lower()
if key not in NUM_NODES:
    raise Exception("Unknown dataset: %s" % opt.dataset)
num_node = NUM_NODES[key]

print(f'Building graphs for: {opt.dataset}')
print(f'num_node: {num_node}')
print(f'Reading from: {seq_file}')
print(f'Window size: {opt.window}')
print(f'Sample num: {opt.sample_num}')

# ── Step 1: Load sequences ────────────────────────────────────────────────────
sequences = pickle.load(open(seq_file, 'rb'))
print(f'Loaded {len(sequences)} training sequences')

# ── Step 2: Build co-occurrence counts ───────────────────────────────────────
adj_dict = defaultdict(lambda: defaultdict(int))

for seq in sequences:
    for i in range(len(seq)):
        for j in range(i + 1, min(i + opt.window + 1, len(seq))):
            u = seq[i]
            v = seq[j]
            if u == v:
                continue
            adj_dict[u][v] += 1
            adj_dict[v][u] += 1

print(f'Built co-occurrence graph with {len(adj_dict)} items')

# ── Step 3: Build list-of-lists format (matches RetailRocket format) ──────────
# adj[i] = list of top-K neighbor item ids
# num[i] = list of corresponding co-occurrence counts
adj_final = [[] for _ in range(num_node)]
num_final = [[] for _ in range(num_node)]

for item_id, neighbors in adj_dict.items():
    if item_id >= num_node:
        continue
    # Sort by frequency descending, keep top sample_num
    sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
    top = sorted_neighbors[:opt.sample_num]
    adj_final[item_id] = [n for n, _ in top]
    num_final[item_id] = [c for _, c in top]

# ── Step 4: Save ─────────────────────────────────────────────────────────────
pickle.dump(adj_final, open(adj_file, 'wb'))
pickle.dump(num_final, open(num_file, 'wb'))

print(f'Saved: {adj_file}')
print(f'Saved: {num_file}')

# ── Step 5: Verify format matches RetailRocket ────────────────────────────────
print('\nVerification:')
print(f'adj type: {type(adj_final)}')
print(f'adj length: {len(adj_final)}')
print(f'adj[0]: {adj_final[0]}')
print(f'adj[1]: {adj_final[1][:5]}...')
print(f'num[1]: {num_final[1][:5]}...')
print('Done!')
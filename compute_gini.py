import pickle
import numpy as np
from collections import Counter

def gini(counts):
    x = np.array(sorted(counts), dtype=float)
    n = len(x)
    cumx = np.cumsum(x)
    return (2 * np.dot(np.arange(1, n+1), x) / (n * cumx[-1])) - (n + 1) / n

for dataset in ['RetailRocket', 'Tmall', 'lastfm']:
    path = f'datasets/{dataset}/train.txt'
    data = pickle.load(open(path, 'rb'))
    seqs, labels = data[0], data[1]
    item_counts = Counter()
    for seq, label in zip(seqs, labels):
        for item in seq:
            item_counts[item] += 1
        item_counts[label] += 1
    counts = list(item_counts.values())
    g = gini(counts)
    print(f"{dataset}: Gini={g:.4f}, unique_items={len(counts)}")

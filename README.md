# CaHTGP

**CaHTGP** for session-based recommendation.

## Requirements

- Python 3.9
- PyTorch 2.8 (CUDA 12.6)
- numpy, scipy, scikit-learn, networkx, tqdm

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

Experiments use three public datasets:

| Dataset      | Source |
|--------------|--------|
| Tmall        | https://tianchi.aliyun.com/dataset/dataDetail?dataId=42 |
| RetailRocket | https://www.kaggle.com/retailrocket/ecommerce-dataset |
| Last.fm      | http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html |

## Project structure

```
├── main.py            # training / evaluation entry point
├── model.py           # CaHTGP model
├── aggregator.py      # graph aggregation layers
├── utils.py           # data loading and helpers
├── build_graphs.py    # builds global item graphs
├── compute_gini.py    # Gini coefficient of recommendations
├── job_*.sh           # SLURM job scripts
└── datasets/          # preprocessed datasets
```

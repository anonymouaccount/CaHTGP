import time, argparse, pickle, os
import numpy as np, torch
from model import *
from utils import *
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def init_seed(seed=2024):
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='retailrocket',
                    help='retailrocket | Tmall | lastfm (any case)')
parser.add_argument('--seed', type=int, default=2024,
                    help='random seed')
parser.add_argument('--dmignn', action='store_true',
                    help='Run in DMI-GNN mode: disable fast scale, '
                         'causal branch, HSIC loss, and fix gate to 0. '
                         'Gives a fair baseline under our protocol.')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_dc', type=float, default=0.1)
parser.add_argument('--lr_dc_step', type=int, default=3)
parser.add_argument('--l2', type=float, default=1e-5)
parser.add_argument('--n_iter', type=int, default=1)
parser.add_argument('--dropout_gcn', type=float, default=None)
parser.add_argument('--dropout_local', type=float, default=0)
parser.add_argument('--dropout_global', type=float, default=None)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--interests', type=int, default=None)
parser.add_argument('--beta', type=float, default=0.005)
parser.add_argument('--length', type=float, default=12)
parser.add_argument('--lambda_ind', type=float, default=0.01)
parser.add_argument('--max_n_neighbor', type=int, default=8)
opt = parser.parse_args()

# Save CLI overrides before config overwrites them
_cli_interests     = opt.interests
_cli_dropout_gcn   = opt.dropout_gcn
_cli_dropout_global = opt.dropout_global

DATASET_ROOT = 'datasets'
REQUIRED_FILES = ['train.txt', 'test.txt', 'adj_12.pkl', 'num_12.pkl']

DATASET_CONFIGS = {
    'retailrocket': dict(
        num_node=36969,
        n_iter=1,
        dropout_local=0.0,
        dropout_global=0.5,
        dropout_gcn=0.8,
        dropout_fast=0.2,
        interests=3,
        length=12,
    ),
    'tmall': dict(
        num_node=40728,
        n_iter=1,
        dropout_local=0.7,
        dropout_global=0.5,
        dropout_gcn=0.5,
        dropout_fast=0.5,
        interests=5,
        length=8,
    ),
    'lastfm': dict(
        num_node=38616,
        n_iter=1,
        dropout_local=0.0,
        dropout_global=0.1,
        dropout_gcn=0,
        dropout_fast=0.1,
        interests=5,
        length=18,
    ),
}

DMIGNN_CONFIGS = {
    'retailrocket': dict(
        num_node=36969,
        n_iter=1,
        dropout_local=0.0,
        dropout_global=0.5,
        dropout_gcn=0.8,
        dropout_fast=0.0,
        interests=3,
        length=12,
    ),
    'tmall': dict(
        num_node=40728,
        n_iter=1,
        dropout_local=0.7,
        dropout_global=1.0,
        dropout_gcn=1.0,
        dropout_fast=0.0,
        interests=5,
        length=8,
    ),
    'lastfm': dict(
        num_node=38616,
        n_iter=1,
        dropout_local=0.0,
        dropout_global=0.1,
        dropout_gcn=0,
        dropout_fast=0.0,
        interests=5,
        length=18,
    ),
}


def resolve_dataset(arg):
    name = os.path.basename(os.path.normpath(arg))
    key = name.lower()
    if key not in DATASET_CONFIGS:
        raise Exception("Unknown Dataset '%s'! Choose one of: %s"
                        % (arg, ', '.join(DATASET_CONFIGS.keys())))
    if os.path.isdir(arg):
        return key, os.path.normpath(arg)
    if os.path.isdir(DATASET_ROOT):
        for folder in os.listdir(DATASET_ROOT):
            full = os.path.join(DATASET_ROOT, folder)
            if os.path.isdir(full) and folder.lower() == key:
                return key, full
        available = [f for f in os.listdir(DATASET_ROOT)
                     if os.path.isdir(os.path.join(DATASET_ROOT, f))]
    else:
        available = []
    raise FileNotFoundError(
        "No folder for dataset '%s' found in '%s/'. Available: %s"
        % (key, DATASET_ROOT, available))


def main():
    init_seed(opt.seed)

    key, data_dir = resolve_dataset(opt.dataset)

    cfg = DMIGNN_CONFIGS[key] if opt.dmignn else DATASET_CONFIGS[key]
    num_node = cfg['num_node']
    for k, v in cfg.items():
        if k != 'num_node':
            setattr(opt, k, v)
    opt.dataset = key

    # CLI overrides: if user passed these explicitly, restore them
    if _cli_interests is not None:
        opt.interests = _cli_interests
    if _cli_dropout_gcn is not None:
        opt.dropout_gcn = _cli_dropout_gcn
    if _cli_dropout_global is not None:
        opt.dropout_global = _cli_dropout_global

    if opt.dmignn:
        opt.lambda_ind = 0.0
        opt.dmignn_mode = True
        print('=' * 60)
        print('MODE: DMI-GNN (fair baseline under our protocol)')
        print('  - Fast scale: DISABLED')
        print('  - Causal branch: DISABLED')
        print('  - HSIC loss: DISABLED')
        print('  - Residual gate: FIXED at 0')
        print('=' * 60)
    else:
        opt.dmignn_mode = False
        print('MODE: CaHTGP (full model)')

    missing = [f for f in REQUIRED_FILES
               if not os.path.isfile(os.path.join(data_dir, f))]
    if missing:
        raise FileNotFoundError(
            "Missing files in '%s': %s" % (data_dir, missing))

    print('Using data from: %s' % data_dir)
    print('Seed: %d' % opt.seed)
    print('interests (K): %d' % opt.interests)
    print('dropout_global: %.2f' % opt.dropout_global)
    print('dropout_gcn: %.2f' % opt.dropout_gcn)

    train_data = pickle.load(open(os.path.join(data_dir, 'train.txt'), 'rb'))
    test_data  = pickle.load(open(os.path.join(data_dir, 'test.txt'),  'rb'))
    adj        = pickle.load(open(os.path.join(data_dir, 'adj_12.pkl'), 'rb'))
    num        = pickle.load(open(os.path.join(data_dir, 'num_12.pkl'), 'rb'))

    train_data = Data(train_data, max_n_neighbor=opt.max_n_neighbor)
    test_data  = Data(test_data,  max_n_neighbor=opt.max_n_neighbor)
    adj, num   = handle_adj(adj, num_node, opt.n_sample_all, num)

    model = trans_to_cuda(CaHTGP(opt, num_node, adj, num))

    print(opt)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    start = time.time()
    best_result = [0, 0, 0]; best_epoch = [0, 0, 0]; bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr, cov = train_test(model, train_data, test_data)
        cov = cov * 100 / (num_node - 1)
        flag = 0
        if hit >= best_result[0]: best_result[0] = hit; best_epoch[0] = epoch; flag = 1
        if mrr >= best_result[1]: best_result[1] = mrr; best_epoch[1] = epoch; flag = 1
        if cov >= best_result[2]: best_result[2] = cov; best_epoch[2] = epoch; flag = 1
        print('Current:\tRecall@20: %.4f\tMRR@20: %.4f\tCov@20: %.4f' % (hit, mrr, cov))
        print('Best:\tRecall@20: %.4f\tMRR@20: %.4f\tCov@20: %.4f\tEpoch: %d, %d, %d' % (
            best_result[0], best_result[1], best_result[2],
            best_epoch[0], best_epoch[1], best_epoch[2]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience: break

    print('-------------------------------------------------------')
    print("Run time: %f s" % (time.time() - start))

if __name__ == '__main__':
    main()
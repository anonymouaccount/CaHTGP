import time, argparse, pickle, os
import numpy as np, torch
from model import *
from utils import *
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def init_seed(seed=2024):
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='retailrocket')
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
parser.add_argument('--dropout_gcn', type=float, default=0)
parser.add_argument('--dropout_local', type=float, default=0)
parser.add_argument('--dropout_global', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--interests', type=int, default=3)
parser.add_argument('--beta', type=float, default=0.005)
parser.add_argument('--length', type=float, default=12)
parser.add_argument('--lambda_ind', type=float, default=0.01)
parser.add_argument('--max_n_neighbor', type=int, default=8)
opt = parser.parse_args()

def main():
    init_seed(2024)
    if opt.dataset == 'retailrocket':
        num_node = 36969
        opt.n_iter = 1
        opt.dropout_local = 0.0       # Session: trusted (user behavior)
        opt.dropout_global = 0.5      # Slow: moderate noise
        opt.dropout_gcn = 0.8
        opt.dropout_fast = 0.2        # Fast: light dropout
        opt.interests = 3
        opt.length = 12
    elif opt.dataset == 'Tmall':
        num_node = 40728
        opt.n_iter = 1
        # Use DMI-GNN's proven config — residual gate handles the rest
        opt.dropout_local = 0.7       # Same as DMI-GNN
        opt.dropout_global = 1.0      # Same as DMI-GNN (slow off during train)
        opt.dropout_gcn = 1.0         # Same as DMI-GNN
        opt.dropout_fast = 0.5        # Fast: moderate (new component)
        opt.interests = 5
        opt.length = 8
    elif opt.dataset == 'lastfm':
        num_node = 38616
        opt.n_iter = 1
        opt.dropout_local = 0.0       # Session: trusted
        opt.dropout_global = 0.1      # Slow: low noise
        opt.dropout_gcn = 0
        opt.dropout_fast = 0.1        # Fast: light
        opt.interests = 5
        opt.length = 18
    else:
        raise Exception('Unknown Dataset!')

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
    adj = pickle.load(open('datasets/' + opt.dataset + '/adj_12.pkl', 'rb'))
    num = pickle.load(open('datasets/' + opt.dataset + '/num_12.pkl', 'rb'))

    train_data = Data(train_data, max_n_neighbor=opt.max_n_neighbor)
    test_data = Data(test_data, max_n_neighbor=opt.max_n_neighbor)
    adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)

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

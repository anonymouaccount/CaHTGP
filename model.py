import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, GlobalAggregator
from torch.nn import Module
import torch.nn.functional as F


# ===========================================================================
# FAST SCALE: GRU Temporal Neighbor Encoder
# ===========================================================================
class FastTemporalEncoder(Module):
    """
    z_fast(t): Event-level temporal dynamics.
    GRU processes neighbor sequence in temporal order,
    attention aggregates based on central node affinity.
    """

    def __init__(self, dim, alpha=0.2):
        super().__init__()
        self.dim = dim
        self.gru = nn.GRU(input_size=2 * dim, hidden_size=dim, batch_first=True)
        self.attn = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LeakyReLU(alpha),
            nn.Linear(dim, 1, bias=False),
        )

    def forward(self, item_embs, neighbor_embs, neighbor_mask):
        """
        item_embs:     (B, N, D)
        neighbor_embs: (B, N, max_nb, D)
        neighbor_mask:  (B, N, max_nb)
        Returns: z_fast (B, N, D)
        """
        B, N, max_nb, D = neighbor_embs.shape

        central_exp = item_embs.unsqueeze(2).expand_as(neighbor_embs)
        gru_input = torch.cat([neighbor_embs, central_exp], dim=-1).view(B * N, max_nb, 2 * D)
        gru_out, _ = self.gru(gru_input)
        gru_out = gru_out.view(B, N, max_nb, D)

        central_exp2 = item_embs.unsqueeze(2).expand(-1, -1, max_nb, -1)
        scores = self.attn(torch.cat([gru_out, central_exp2], dim=-1)).squeeze(-1)
        scores = scores.masked_fill(neighbor_mask == 0, -1e9)
        beta = F.softmax(scores, dim=-1)

        z_fast = torch.sum(beta.unsqueeze(-1) * neighbor_embs, dim=2)

        has_nb = (neighbor_mask.sum(dim=-1, keepdim=True) > 0).float()
        z_fast = z_fast * has_nb + item_embs * (1 - has_nb)
        return z_fast


# ===========================================================================
# CAUSAL EXTRACTOR (for Slow scale only)
# ===========================================================================
class CausalExtractor(Module):
    """h_causal = h_factual - sigmoid(alpha) * Proj(h_cf)"""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, h_factual, h_spurious):
        return h_factual - torch.sigmoid(self.alpha) * self.proj(h_spurious)


class IndependenceLoss(Module):
    """HSIC: cross-correlation(h_causal, h_spurious) → 0"""

    def forward(self, h_causal, h_spurious, mask):
        mask_f = mask.float().unsqueeze(-1)
        n_valid = mask_f.sum(dim=1, keepdim=True).clamp(min=1)
        c = F.normalize((h_causal * mask_f).sum(dim=1) / n_valid.squeeze(-1), dim=-1)
        s = F.normalize((h_spurious * mask_f).sum(dim=1) / n_valid.squeeze(-1), dim=-1)
        B = c.shape[0]
        return torch.mean((torch.mm(c.T, s) / B) ** 2)


# ===========================================================================
# CaHTGP MAIN MODEL
# ===========================================================================
class CaHTGP(Module):

    def __init__(self, opt, num_node, adj_all, num):
        super(CaHTGP, self).__init__()
        self.opt = opt
        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.dropout_fast = getattr(opt, 'dropout_fast', 0.2)
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.K = opt.interests
        self.length = opt.length
        self.beta_div = opt.beta
        self.lambda_ind = opt.lambda_ind

        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num_all = trans_to_cuda(torch.Tensor(num)).float()

        # Shared embeddings
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        # === FAST SCALE: Temporal neighbor GRU ===
        self.fast_encoder = FastTemporalEncoder(self.dim, alpha=opt.alpha)

        # === SESSION SCALE: LocalAgg (from DMI-GNN) ===
        self.local_agg = LocalAggregator(self.dim, opt.alpha, dropout=0.0)

        # === SLOW SCALE: Factual GlobalAgg (from DMI-GNN) ===
        self.global_agg = []
        for i in range(self.hop):
            act = torch.relu if opt.activate == 'relu' else torch.tanh
            agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=act)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        # === SLOW SCALE CF: Counterfactual GlobalAgg (from CauMI-GNN) ===
        self.cf_global_agg = []
        for i in range(self.hop):
            act = torch.relu if opt.activate == 'relu' else torch.tanh
            agg = GlobalAggregator(self.dim, opt.dropout_gcn, act=act)
            self.add_module('cf_agg_gcn_{}'.format(i), agg)
            self.cf_global_agg.append(agg)

        # === CAUSAL EXTRACTOR (for slow scale) ===
        self.causal_extractor = CausalExtractor(self.dim)
        self.independence_loss = IndependenceLoss()

        # === RESIDUAL GATE: start from DMI-GNN, gradually add new components ===
        # Init at -3.0 → sigmoid(-3) ≈ 0.05 → nearly pure DMI-GNN at start
        self.residual_gate = nn.Parameter(torch.tensor(-3.0))

        # === MULTI-INTEREST HEAD (from DMI-GNN, unchanged) ===
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, self.K))
        self.glu1 = nn.Linear(self.dim, self.K * self.dim)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc,
        )
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):
        return self.adj_all[target.view(-1)], self.num_all[target.view(-1)]

    def _run_global_branch(self, inputs, mask_item, item, global_aggs, debias=False):
        """Run one global aggregation branch (factual or CF)."""
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]

        item_neighbors = [inputs]
        weight_neighbors = []
        support_size = seqs_len

        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))

        entity_vectors = [self.embedding(i) for i in item_neighbors]

        if debias:
            weight_vectors = []
            for w in weight_neighbors:
                inv_w = 1.0 / (w + 1.0)
                inv_w = inv_w / inv_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                weight_vectors.append(inv_w)
        else:
            weight_vectors = weight_neighbors

        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)
        sum_item_emb = sum_item_emb.unsqueeze(-2)
        session_info = [sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1) for i in range(self.hop)]

        for n_hop in range(self.hop):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            for hop in range(self.hop - n_hop):
                vector = global_aggs[n_hop](
                    self_vectors=entity_vectors[hop],
                    neighbor_vector=entity_vectors[hop + 1].view(shape),
                    masks=None, batch_size=batch_size,
                    neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num),
                    extra_vector=session_info[hop],
                )
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        return entity_vectors[0].view(batch_size, seqs_len, self.dim)

    def compute_scores(self, hidden, z_slow_cf, mask):
        """Multi-interest scoring + diversity + independence losses."""
        mask_f = mask.float().unsqueeze(-1)
        B, L, D = hidden.shape

        pos_emb = self.pos_embedding.weight[:L].unsqueeze(0).repeat(B, 1, 1)

        # Multi-interest extraction (DMI-GNN)
        nh = torch.tanh(torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1))
        nh = torch.sigmoid(self.glu1(nh))
        nh_split = torch.split(nh, self.dim, dim=2)
        nh = torch.stack(nh_split, dim=3)

        w2 = self.w_2.unsqueeze(0).repeat(B, L, 1, 1)
        beta = torch.sum(nh * w2, dim=2) * mask_f.expand(-1, -1, self.K)

        # Diversity loss (DMI-GNN original)
        dims = list(range(self.K))
        nbeta = torch.empty_like(beta, device=beta.device)
        for i in dims:
            nbeta[:, :, i] = F.normalize(beta[:, :, i], p=2, dim=1)
        sumask = torch.sum(mask_f.expand(-1, -1, self.K), 1)
        lens = sumask[:, 0] - self.length
        sim_loss = torch.zeros(B, dtype=torch.float32, device=beta.device)
        for i in dims[:-1]:
            for j in dims[i + 1:]:
                sim_loss += torch.abs(torch.sum(nbeta[:, :, i] * nbeta[:, :, j], dim=1))
        sim_loss = sim_loss * 2 / (self.K * (self.K - 1))
        div_loss = torch.sum(torch.sigmoid(sim_loss * lens))

        # Build K interests
        selects = [torch.sum(beta[:, :, k].unsqueeze(-1) * hidden, 1) for k in dims]
        select = torch.stack(selects, dim=0)

        # Score: L2 norm + hard MAX
        b = F.normalize(self.embedding.weight[1:], p=2.0, dim=-1)
        scores = torch.matmul(select, b.transpose(1, 0))
        max_scores, _ = torch.max(scores, dim=0)
        sum_scores = torch.sum(scores, dim=0)

        # Independence loss (causal ⊥ spurious on slow scale)
        ind_loss = self.independence_loss(hidden, z_slow_cf, mask)

        aux_loss = self.beta_div * div_loss + self.lambda_ind * ind_loss
        return max_scores, aux_loss, sum_scores

    def forward(self, inputs, adj, mask_item, item, neighbor_seqs, neighbor_masks):
        """
        Three-scale temporal encoding + causal debiasing on slow scale.

        inputs:         (B, N) unique items in session (node space)
        adj:            (B, N, N) local adjacency matrix
        mask_item:      (B, N) valid item mask
        item:           (B, N) = inputs (used for global aggregation)
        neighbor_seqs:  (B, N, max_nb) temporal neighbor IDs (node space)
        neighbor_masks: (B, N, max_nb) neighbor valid mask
        """
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]

        # Item embeddings in NODE SPACE (unique items)
        h = self.embedding(inputs)       # (B, N, D)
        h = F.normalize(h, p=2.0, dim=-1)

        # === FAST SCALE: Temporal GRU on neighbors (node space) ===
        nb_embs = self.embedding(neighbor_seqs)  # (B, N, max_nb, D)
        z_fast = self.fast_encoder(h, nb_embs, neighbor_masks)  # (B, N, D)
        z_fast = F.dropout(z_fast, self.dropout_fast, training=self.training)

        # === SESSION SCALE: LocalAgg (node space) ===
        z_session = self.local_agg(h, adj, mask_item)  # (B, N, D)
        z_session = F.dropout(z_session, self.dropout_local, training=self.training)

        # === SLOW SCALE: Factual GlobalAgg (sequence space → node space) ===
        z_slow = self._run_global_branch(item, mask_item, item, self.global_agg, debias=False)
        z_slow = F.dropout(z_slow, self.dropout_global, training=self.training)

        # === SLOW SCALE CF: Counterfactual GlobalAgg ===
        z_slow_cf = self._run_global_branch(item, mask_item, item, self.cf_global_agg, debias=True)
        z_slow_cf = F.dropout(z_slow_cf, self.dropout_global, training=self.training)

        # === CAUSAL EXTRACTION on slow scale ===
        z_slow_causal = self.causal_extractor(z_slow, z_slow_cf)

        # === RESIDUAL GATED FUSION ===
        # Base = DMI-GNN equivalent: session(local) + slow(global)
        h_base = z_session + z_slow

        # New contributions from CaHTGP:
        #   fast temporal dynamics + causal correction (debiasing delta)
        h_new = z_fast + (z_slow_causal - z_slow)

        # Learned gate: starts near 0 (=DMI-GNN), opens if helpful
        gate = torch.sigmoid(self.residual_gate)
        output = h_base + gate * h_new

        return output, z_slow_cf


# ===========================================================================
# Helpers + Training
# ===========================================================================
def trans_to_cuda(v):
    return v.cuda() if torch.cuda.is_available() else v

def trans_to_cpu(v):
    return v.cpu() if torch.cuda.is_available() else v


def forward(model, data):
    alias_inputs, adj, items, mask, targets, inputs, nb_seqs, nb_masks = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()
    nb_seqs = trans_to_cuda(nb_seqs).long()
    nb_masks = trans_to_cuda(nb_masks).float()

    # Model operates in NODE SPACE (unique items)
    # items = unique nodes, inputs = raw reversed sequence
    hidden, z_slow_cf = model(items, adj, mask, items, nb_seqs, nb_masks)

    # Map from node space back to sequence space via alias_inputs
    get = lambda i: hidden[i][alias_inputs[i]]
    get_cf = lambda i: z_slow_cf[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    seq_cf = torch.stack([get_cf(i) for i in torch.arange(len(alias_inputs)).long()])
    return targets, model.compute_scores(seq_hidden, seq_cf, mask)


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(
        train_data, num_workers=4, batch_size=model.batch_size,
        shuffle=True, pin_memory=True,
    )
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, (scores, aux_loss, _) = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1) + aux_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(
        test_data, num_workers=4, batch_size=model.batch_size,
        shuffle=False, pin_memory=True,
    )
    hit, mrr = [], []
    cov = set()
    for data in test_loader:
        targets, (scores, _, _) = forward(model, data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = targets.numpy()
        for score, target, m in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            cov.update(score)
            pos = np.where(score == target - 1)[0]
            mrr.append(0 if len(pos) == 0 else 1 / (pos[0] + 1))
    return [np.mean(hit) * 100, np.mean(mrr) * 100, len(cov)]

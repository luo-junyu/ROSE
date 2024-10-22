import torch
import random
import numpy as np
import logging
import time
import os
import networkx as nx


def compute_kernel_batch(x):
    batch_size = x.size(0)
    num_aug = x.size(1)
    dim = x.size(2)
    n_samples = batch_size * num_aug

    y = x.clone()
    x = x.unsqueeze(1).unsqueeze(3)  # (B, 1, n, 1, d)
    y = y.unsqueeze(0).unsqueeze(2)  # (1, B, 1, n, d)
    tiled_x = x.expand(batch_size, batch_size, num_aug, num_aug, dim)
    tiled_y = y.expand(batch_size, batch_size, num_aug, num_aug, dim)

    L2_distance = (tiled_x - tiled_y).pow(2).sum(-1)
    bandwidth = torch.sum(L2_distance.detach()) / (n_samples ** 2 - n_samples)

    return torch.exp(-L2_distance / bandwidth)


def compute_mmd_batch(x):
    batch_size = x.size(0)
    batch_kernel = compute_kernel_batch(x)  # B*B*n*n
    batch_kernel_mean = batch_kernel.reshape(batch_size, batch_size, -1).mean(2)  # B*B
    self_kernel = torch.diag(batch_kernel_mean)
    x_kernel = self_kernel.unsqueeze(1).expand(batch_size, batch_size)
    y_kernel = self_kernel.unsqueeze(0).expand(batch_size, batch_size)
    mmd = x_kernel + y_kernel - 2*batch_kernel_mean

    return mmd.detach()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def create_mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


def get_logger(args):
    create_mkdir(args.log_dir)
    log_path = os.path.join(args.log_dir, args.DS+'_'+args.log_file)
    print('logging into %s' % log_path)

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info('#' * 20)

    # record arguments
    args_str = ""
    for k, v in sorted(vars(args).items()):
        args_str += "%s" % k + "=" + "%s" % v + "; "
    logger.info(args_str)
    print(args_str)
    logger.info("DS: %s" % args.DS)
    logger.info(f'Split: {args.data_split}, Source Index: {args.source_index}, Target Index: {args.target_index}')

    return logger

def neighborhood(G, node, n):
    paths = nx.single_source_shortest_path(G, node)
    return [node for node, traversed_nodes in paths.items()
            if len(traversed_nodes) == n+1]

def save_model(ckpt_dir, model):
    saved_state = {
        'model': model.state_dict(),
    }
    torch.save(saved_state, ckpt_dir)

def generate_sub_features_idx(adj_batch, features_batch, size_subgraph = 10, k_neighbor=1):
    sub_features_idx_list, sub_adj_list = [],[]
    for i in range(len(adj_batch)):
        adj = adj_batch[i]
        x = features_batch[i]
        num_B_nodes = x.shape[0]
        G = nx.from_numpy_array(adj.to_dense().cpu().numpy())
        subgraph_idx = []
        x_sub_adj = torch.zeros(x.shape[0], size_subgraph, size_subgraph)
        x_sub_idx = torch.zeros(x.shape[0], size_subgraph)

        for node in range(x.shape[0]):

            # determine neighbors' idx
            tmp = []
            for k in range(k_neighbor+1):
                tmp = tmp + neighborhood(G, node, k)
            if len(tmp) > size_subgraph:
                tmp = tmp[:size_subgraph]
            sub_idxs = tmp
            
            if len(tmp) < size_subgraph:
                padded_sub_idxs = tmp + [num_B_nodes for i in range(size_subgraph-len(tmp))]
            else:
                padded_sub_idxs = tmp
     
            x_sub_idx[node] = torch.tensor(padded_sub_idxs)

            # corresponding neighbor and neighbor features      
            G_sub = G.subgraph(sub_idxs)
            tmp = nx.to_numpy_array(G_sub)
            if tmp.shape[0] < size_subgraph:
                tmp_adj = np.zeros([size_subgraph, size_subgraph])
                tmp_adj[:tmp.shape[0],:tmp.shape[1]] = tmp
                tmp = tmp_adj
            x_sub_adj_ = torch.from_numpy(tmp).float()
            if 2 in x_sub_adj_:
                x_sub_adj_ = x_sub_adj_/2
            x_sub_adj[node] = x_sub_adj_

        sub_features_idx_list.append(x_sub_idx.long()) 
        sub_adj_list.append(x_sub_adj)

    return sub_adj_list, sub_features_idx_list
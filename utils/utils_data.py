from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
import torch_geometric.transforms as T
from torch.utils.data import Dataset
import torch
from .data_splits import get_splits_in_domain, get_domain_splits
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
import random
from torch_geometric.data import DataLoader
from utils.utils import generate_sub_features_idx


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def kernel_process(data):
    adj_matrix = torch.sparse_coo_tensor(data.edge_index, torch.ones(data.edge_index.shape[1]),
            (data.x.shape[0], data.x.shape[0]))
    adj, idx = generate_sub_features_idx([adj_matrix], [data.x], size_subgraph=10, k_neighbor=1)
    data.ker_adj = adj[0]
    data.ker_idx = idx[0]
    print('process kernel!')
    return data

def get_dataset(DS, path, args):
    setup_seed(0)
    
    dataset = TUDataset(path, name=DS, use_node_attr=True, pre_transform=kernel_process)
    print(f'Dataset: {DS}, Length: {len(dataset)}')
    source_split_index = args.source_index
    target_split_index = args.target_index
    split = args.data_split
    split_dataset = get_domain_splits(dataset, split)
    source_dataset = split_dataset[source_split_index]
    target_dataset = split_dataset[target_split_index]
    source_train_dataset, source_val_dataset = get_splits_in_domain(source_dataset)
    target_train_dataset, target_test_dataset = get_splits_in_domain(target_dataset)

    return dataset, (source_train_dataset, source_val_dataset, target_train_dataset, target_test_dataset)


def split_confident_data(model, target_train_dataset, device, good_percentage=0.6, confident_percentage=0.6): 
    # good_confident_dataset, fair_confident_dataset, inconfident_dataset, confident_dataset = target_train_dataset, target_train_dataset, target_train_dataset, target_train_dataset

    model.eval()
    significance_scores = []

    new_target_train_dataset = []
    
    for idx in range(len(target_train_dataset)):
        data = target_train_dataset[idx].to(device)
        data.batch = torch.tensor([0] * data.num_nodes, device=device)  # Create a batch tensor with all nodes belonging to the same graph
        data.num_graphs = 1  # Set number of graphs to 1
        # Forward pass to obtain projections and predictions
        x, x_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
        pred = model.classifier(x_proj)
        pseudo_label = pred.argmax(dim=-1).detach().cpu()
        new_target_train_dataset.append(target_train_dataset[idx].update({'pseudo_label': pseudo_label}))

        # Find neighbors and calculate significance scores
        nei = find_top_k_neighbors(x_proj.detach().cpu().numpy(), k=10)
        purities = neighbor_purity(pred.detach().cpu().numpy(), nei)
        affinities = neighbor_affinity(x_proj.detach().cpu().numpy(), nei)
        
        for purity, affinity in zip(purities, affinities):
            significance_scores.append(purity * affinity)

    target_train_dataset = new_target_train_dataset

    sorted_indices = np.argsort(-np.array(significance_scores))
    num_confident = int(confident_percentage * len(target_train_dataset))
    num_good_confident = int(good_percentage * len(target_train_dataset))

    confident_indices = sorted_indices[:num_confident].tolist()
    inconfident_indices = sorted_indices[num_confident:].tolist()
    good_confident_indices = sorted_indices[:num_good_confident].tolist()
    fair_confident_indices = sorted_indices[num_good_confident:].tolist()

    inconfident_dataset = [target_train_dataset[i] for i in inconfident_indices]
    confident_dataset = [target_train_dataset[i] for i in confident_indices]
    good_confident_dataset = [target_train_dataset[i] for i in good_confident_indices]
    fair_confident_dataset = [target_train_dataset[i] for i in fair_confident_indices]

    return good_confident_dataset, fair_confident_dataset, inconfident_dataset, confident_dataset


def neighbor_purity(probability_distributions, neighbors_indices):
    purities = []
    
    for i, neighbors in enumerate(neighbors_indices):
        neighbor_probs = probability_distributions[neighbors]
        mean_prob_dist = np.mean(neighbor_probs, axis=1)
        
        if np.any(np.isnan(mean_prob_dist)) or np.any(np.isinf(mean_prob_dist)):
            print(f"Warning: NaN or Inf encountered in mean_prob_dist at index {i}.")
            purities.append(float('nan'))
            continue

        mean_prob_dist = mean_prob_dist / (np.sum(mean_prob_dist) + 1e-10)
        entropy = -np.sum(mean_prob_dist * np.log(mean_prob_dist + 1e-10))
        
        if np.isnan(entropy):
            breakpoint()
            print(f"Warning: Entropy calculation resulted in NaN at index {i}.")
            entropy = 0.0
        
        purities.append(entropy)
    
    return purities

def neighbor_affinity(representations, neighbors_indices):
    affinities = []
    
    for i, neighbors in enumerate(neighbors_indices):
        sample_rep = representations[i]
        neighbor_reps = representations[neighbors]
        similarities = cosine_similarity([sample_rep], neighbor_reps)[0]
        affinity = np.mean(similarities)
        affinities.append(affinity)
    
    return affinities

def find_top_k_neighbors(representations, k=10):
    # Calculate the pairwise Euclidean distances between all samples
    distances = euclidean_distances(representations)
    
    # For each sample, find the indices of the k nearest neighbors
    neighbors_indices = []
    for i in range(distances.shape[0]):
        # Exclude the distance to itself by setting it to a large number (infinity)
        distances[i, i] = np.inf
        # Get the indices of the top-k nearest neighbors (smallest distances)
        nearest_neighbors = np.argsort(distances[i])[:k]
        neighbors_indices.append(nearest_neighbors.tolist())
    
    return neighbors_indices

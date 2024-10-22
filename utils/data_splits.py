import torch

def get_splits_in_domain(dataset, ratio=0.8):
    data_idx = torch.randperm(len(dataset))
    dataset = dataset[data_idx]
    num_data = len(data_idx)
    num_train = int(num_data * ratio)
    train_index = data_idx[:num_train]
    test_index = data_idx[num_train:]

    train_dataset = dataset[train_index]
    test_dataset = dataset[test_index]

    return train_dataset, test_dataset


def get_domain_splits(dataset, split=4):
    node_density = []

    for i in range(len(dataset)):
        node_density.append(dataset[i].num_edges / (dataset[i].num_nodes * (dataset[i].num_nodes - 1)))
    node_density = torch.tensor(node_density)
    node_density, data_idx = torch.sort(node_density, descending=False)

    return_dataset = []
    for i in range(split):
        return_dataset.append(dataset[data_idx[i*(len(dataset) // split) : (i+1)*(len(dataset) // split)]])

    return return_dataset

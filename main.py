import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader
import time
from arguments import arg_parse
from utils.utils_data import get_dataset, split_confident_data
from utils.graph_aug import AugTransform
from models.model import GNN
from models.kergnn import kergnn
from utils.utils import get_logger, setup_seed, save_model
import copy
from tqdm import tqdm


@torch.no_grad()
def test(loader, model):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        x, x_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
        pred = model.classifier(x_proj).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


@torch.no_grad()
def eval_train(loader, model):
    model.eval()

    total_correct = 0
    for data_dict in loader:
        data = data_dict.to(device)
        x, x_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
        pred = model.classifier(x_proj).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


def run(seed=0):
    epochs = args.epochs
    eval_interval = args.eval_interval
    log_interval = args.log_interval
    batch_size = args.batch_size
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    source_save_path = osp.join('.', 'ckpt', f'{DS}-{args.source_index}.pth')
    target_save_path = osp.join('.', 'ckpt', f'{DS}-{args.source_index}-{args.target_index}.pth')

    dataset, (source_train_dataset, source_val_dataset, target_train_dataset, target_test_dataset) = get_dataset(DS, path, args)

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    source_train_loader = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    source_val_loader = DataLoader(source_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    train_transforms = AugTransform(args.aug)
    print('Calculating uniform targets...')
    criterion = nn.CrossEntropyLoss()

    # print(len(dataset))
    dataset_num_features = source_train_dataset[0].x.shape[1]
    print(f'num_features: {dataset_num_features}')
    setup_seed(seed)

    model = GNN(dataset_num_features, args.hidden_dim, args.num_gc_layers, dataset.num_classes, args, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    best_val_acc = 0.0
    final_test_acc = 0.0
    model_for_tta = None

    for epoch in tqdm(range(1, epochs + 1)):
        time_start = time.time()
        loss_all = 0
        model.train()

        dataloader = source_train_loader

        # Train on Source Domain 
        for data_dict in dataloader:
            data = data_dict.to(device)
            optimizer.zero_grad()
            x, x_proj = model(data.x, data.edge_index, data.batch, data.num_graphs)
            pred = model.classifier(x_proj)
            loss = criterion(pred, data.y)

            loss.backward()

            loss_all += loss.item() * data.num_graphs
            optimizer.step()

        if epoch % eval_interval == 0:
            model.eval()
            val_acc = test(source_val_loader, model)
            if val_acc >= best_val_acc:
                model_for_tta = copy.deepcopy(model)

    # Source Free Domain Adaptation

    model = model_for_tta

    with torch.no_grad():
        good_confident_dataset, fair_confident_dataset, inconfident_dataset, confident_dataset = \
            split_confident_data(model,target_train_dataset,device)

    good_confident_dataloader = DataLoader(good_confident_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    fair_confident_dataloader = DataLoader(fair_confident_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    confident_dataloader = DataLoader(confident_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    inconfident_dataloader = DataLoader(inconfident_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # print(f'good_confident_dataset:{len(good_confident_dataset)},inconfident_dataset:{len(inconfident_dataset)}')

    target_epochs = args.target_epochs
    best_val_acc = 0.0
    final_test_acc = 0.0

    if args.use_kergnn:
        kg_model = kergnn(dataset_num_features, dataset.num_classes, hidden_dims=[0,16,32], kernel='drw', max_step=1, 
            num_mlp_layers=1, mlp_hidden_dim=16, dropout_rate=0.4,
            size_graph_filter=[4,4], size_subgraph=10, no_norm=True).to(device)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(kg_model.parameters()), lr=args.sfda_lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.sfda_lr)
    
    if args.use_kergnn:
        for epoch in tqdm(range(1, 10)):
            loss_all = 0
            model.train()
            for data_dict in confident_dataloader:
                data = data_dict.to(device)
                optimizer.zero_grad()
                ker_x = kg_model([data.ker_adj], data.x, [data.ker_idx], data.batch)
                loss = criterion(ker_x, data.pseudo_label)
                loss.backward()
                optimizer.step()
                loss_all += loss.item() * data.num_graphs

            model.eval()
            train_acc = eval_train(confident_dataloader, model)
            val_acc = test(fair_confident_dataloader, model)
            test_acc = test(target_test_loader, model)
            tqdm.write(f'[KerGNN] Epoch: {epoch:03d}, Loss: {loss_all / len(confident_dataloader):.2f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                f'Test: {test_acc:.4f}')

    for epoch in tqdm(range(1, target_epochs + 1)):
        loss_all = 0
        model.train()
        if args.use_kergnn:
            kg_model.train()
        if args.use_unconf_consist:
            inconfident_dataloader_iter = iter(inconfident_dataloader)

        for data_good, data_fair in zip(good_confident_dataloader, fair_confident_dataloader):
            
            data_good = data_good.to(device)
            data_fair = data_fair.to(device)
            
            model_retain = copy.deepcopy(model)

            optimizer.zero_grad()
            loss = 0
            pseudo_loss = 0
            consist_loss = 0

            x, x_proj = model(data_good.x, data_good.edge_index, data_good.batch, data_good.num_graphs)
            pred = model.classifier(x_proj)
            pseudo_label_gcn = pred.argmax(dim=-1)
            pseudo_label = pseudo_label_gcn.detach()
            
            if args.use_kergnn:
                with torch.no_grad():
                    ker_x = kg_model([data_good.ker_adj], data_good.x, [data_good.ker_idx], data_good.batch)
                    pseudo_label_ker = ker_x.argmax(dim=-1)
                    pseudo_label_ker.detach()
            
            if args.use_cps and args.use_kergnn:
                pseudo_loss += criterion(pred, pseudo_label_ker) * 0.01
                pseudo_loss += criterion(ker_x, pseudo_label) * 0.01
            else:
                pseudo_loss += criterion(pred, pseudo_label)
                if args.use_kergnn:
                    pseudo_loss += criterion(ker_x, pseudo_label)

            if args.use_unconf_consist:
                data_inconfident = next(inconfident_dataloader_iter).to(device)
                x, x_proj = model(data_inconfident.x, data_inconfident.edge_index, data_inconfident.batch, data_inconfident.num_graphs)
                pred = model.classifier(x_proj)
                ker_x = kg_model([data_inconfident.ker_adj], data_inconfident.x, [data_inconfident.ker_idx], data_inconfident.batch)
                consist_loss += F.mse_loss(pred, ker_x) * 0.01
            
            if args.use_meta:
                # print(f'pseudo_loss: {pseudo_loss} consist_loss: {consist_loss}')
                loss = pseudo_loss + consist_loss
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
                x, x_proj = model(data_fair.x, data_fair.edge_index, data_fair.batch, data_fair.num_graphs)
                pred = model.classifier(x_proj)
                loss_1 = criterion(pred, data_fair.pseudo_label)
                grad_ret_1 = torch.autograd.grad(loss_1, model.parameters(), allow_unused=True)

                x, x_proj = model_retain(data_fair.x, data_fair.edge_index, data_fair.batch, data_fair.num_graphs)
                pred = model_retain.classifier(x_proj)
                loss_2 = criterion(pred, data_fair.pseudo_label)
                grad_ret_2 = torch.autograd.grad(loss_2, model_retain.parameters(), allow_unused=True)

                meta_grad = grad_ret_1 + grad_ret_2
                # meta_loss = loss_1 + loss_2

                for param, grad in zip(model.parameters(), meta_grad):
                    if not grad is None:
                        param.grad = grad * 0.01
                optimizer.step()
                optimizer.zero_grad()
            else:
                loss = pseudo_loss + consist_loss
                loss.backward()
                optimizer.step()

            loss_all += loss.item() * data_good.num_graphs            
            optimizer.step()


        if epoch % 2 == 0:
            model.eval()
            test_acc = test(target_test_loader, model)
            tqdm.write(f'SFDA Epoch: {epoch:03d}, Loss: {loss_all / len(dataloader):.2f}, Test: {test_acc:.4f}')
    


if __name__ == '__main__':
    args = arg_parse()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = get_logger(args)
    run()


import pandas as pd
import torch.nn as nn
import time
import numpy as np
import os
import torch
import argparse

from torch_geometric.loader import DataLoader
from models.gcn_transformer import GCN_Trans
from utils import set_seed
from utils import get_metrics
from create_data import read_raw_data_loop_ligand
from create_data import trans

def load_data(args):
    dataset = args['dataset']
    batch_size = args['batch_size']
    seed = args['seed']
    k_value = args['k_value']
    max_seq_len = args['max_seq_len']  
    n_splits = args['n_splits']
    val_size = args['val_size']  # 使用tes_size作为验证集和测试集的比例
    
    # 读取所有折的数据
    if dataset == 'loop-ligand':
        all_folds = read_raw_data_loop_ligand(dataset, n_splits, seed, val_size)
    
    data_loaders = []
    for fold, (df_tra, df_val, df_tes) in enumerate(all_folds, start=1):
        tra_data = trans(dataset, df_tra, 'tra', seed, k_value, max_seq_len, fold)
        val_data = trans(dataset, df_val, 'val', seed, k_value, max_seq_len, fold)
        tes_data = trans(dataset, df_tes, 'tes', seed, k_value, max_seq_len, fold)
        
        # 创建数据加载器
        loader_tra = DataLoader(tra_data, batch_size=batch_size, shuffle=True)
        loader_val = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        loader_tes = DataLoader(tes_data, batch_size=batch_size, shuffle=False)
        
        data_loaders.append((loader_tra, loader_val, loader_tes))
    
    return data_loaders

# 新增计算正样本权重的函数
def compute_pos_weight(loader):
    """Calculate pos_weight from training loader"""
    all_labels = []
    # 遍历整个训练数据集收集标签
    for data in loader.dataset:
        all_labels.append(data.y.item())
    class_counts = np.bincount(all_labels)
    # 添加平滑因子防止除零
    pos_weight = torch.tensor([(class_counts[0]+1e-7)/(class_counts[1]+1e-7)], 
                            dtype=torch.float32)
    return pos_weight


def training(model, device, loader_tra, loss_fn, optimizer):
    model.train()
    
    total_loss = 0
    y_pred_tra = []
    y_true_tra = []
    
    for batch_idx, data in enumerate(loader_tra):
        data = data.to(device)
        target = data.y.view(-1, 1).float().to(device)
        
        optimizer.zero_grad()
        output = model(data)
        out = output['out']
        loss_aux = output['loss_aux']

        # 计算损失
        loss = loss_fn(out, target)
        if loss_aux is not None:
            loss = (1 - args['aux_weight']) * loss + args['aux_weight'] * loss_aux
        
        # 反向传播和优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(target)

        # accumulatte predictions and labels
        pred = torch.sigmoid(out).detach().cpu().view(-1).numpy().tolist()
        y_pred_tra += pred
        
        label = data.y.view(-1).detach().cpu().numpy().tolist()
        y_true_tra += label

    loss_value_tra = round(total_loss / len(y_pred_tra), 5)
    
    metrics_tra = get_metrics(real_score=y_true_tra, predict_score=y_pred_tra)

    return loss_value_tra, y_pred_tra, y_true_tra, metrics_tra


def predicting(model, device, loader, loss_fn):
    model.eval()
    
    total_loss = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            target = data.y.view(-1, 1).float().to(device)
            
            output = model(data)
            out = output['out']
            loss_aux = output['loss_aux']
            
            loss = loss_fn(out, target)
            if loss_aux is not None:
                loss = (1 - args['aux_weight']) * loss + args['aux_weight'] * loss_aux
                
            total_loss += loss.item() * len(target)
            
            pred = torch.sigmoid(out).detach().cpu().view(-1).numpy().tolist()
            y_pred += pred
            
            label = data.y.view(-1).detach().cpu().numpy().tolist()
            y_true += label
    loss_value = round(total_loss / len(y_pred), 5)
    metrics_val_tes = get_metrics(real_score=y_true, predict_score=y_pred)
    return loss_value, np.array(y_pred), np.array(y_true), metrics_val_tes


def TraValTes(args, data_loaders):
    dataset = args['dataset']
    model_name = args['model_name']
    lr = args['lr']
    device = args['device']
    time_str = args['time_str']
    epochs = args['epochs']
    
    model_dict = {'GCN_Trans': GCN_Trans}
    

    
    model_obj = model_dict[model_name]
    model = model_obj(args)
    model = model.to(device)
    
    loss_fn = nn.BCEWithLogitsLoss().to(device)
   
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           mode='min', 
                                                           patience=3, 
                                                           factor=0.5)
    
    for fold, (loader_tra, loader_val, loader_tes) in enumerate(data_loaders, start=1):
        
        if fold > 1:    ######只运行第一折######
           break
        
        print(f"Processing {model_name} fold {fold}")

        pos_weight = compute_pos_weight(loader_tra).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

        best_score = 0
        result_metrics = np.zeros((epochs, 1 + 7 * 3))

        for epoch in range(epochs):
            start_time = time.time()

            out_tra = training(model, device, loader_tra, loss_fn, optimizer)
            loss_value_tra, y_pred_tra, y_true_tra, metrics_tra = out_tra

            out_val = predicting(model, device, loader_val, loss_fn)
            loss_value_val, y_pred_val, y_true_val, metrics_val = out_val

            out_tes = predicting(model, device, loader_tes, loss_fn)
            loss_value_tes, y_pred_tes, y_true_tes, metrics_tes = out_tes

            scheduler.step(loss_value_val)

            result_metrics[epoch, 0] = epoch
            for tmp_i, (s_tra, s_val, s_tes) in enumerate(
                zip(metrics_tra, metrics_val, metrics_tes)):
                result_metrics[epoch, 3 * tmp_i + 1] = s_tra
                result_metrics[epoch, 3 * tmp_i + 2] = s_val
                result_metrics[epoch, 3 * tmp_i + 3] = s_tes

            end_time = time.time()
            time_elapsed = round((end_time - start_time) / 60, 1)
            print(f'---epoch:{epoch}---\nelapsed minutes: {time_elapsed}')
            print(f'---Tra---{loss_value_tra}---{metrics_tra}\n')
            print(f'---Val---{loss_value_val}---{metrics_val}\n')
            print(f'---Tes---{loss_value_tes}---{metrics_tes}\n\n')
            if metrics_val[0] > best_score:
                best_score = metrics_val[0]
                best_epoch = epoch
                metrics_tra_best = metrics_tra
                metrics_val_best = metrics_val
                metrics_tes_best = metrics_tes

                model_folder = "json"
                if not os.path.exists(model_folder):
                    os.makedirs(model_folder)
                model_file_name = os.path.join(model_folder, f'{model_name}_{dataset}_fold{fold}.json')
                torch.save(model.state_dict(), model_file_name)
                print(f'---best Tes---{metrics_tes_best}\n\n')

        metrics_names = ['AUC', 'AUPR', 'F1_score', 'Accuracy',
                         'Recall', 'Specificity', 'Precision']
        columns = ['Epoch'] + [f'{prefix}_{metric}'
                               for metric in metrics_names
                               for prefix in ['Tra', 'Val', 'Tes']]
        results_df = pd.DataFrame(result_metrics, columns=columns)

        result_folder = "result"
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        results_df.to_csv(f'{result_folder}/result_metrics_{model_name}_{dataset}_fold{fold}_{time_str}.csv', index=False)

        print(f"Best epoch: {best_epoch}")
        print(f"Best Tra : {metrics_tra_best}")
        print(f"Best Val : {metrics_val_best}")
        print(f"Best Tes : {metrics_tes_best}")

    return


def get_args():
    parser = argparse.ArgumentParser()

    # 数据集和模型参数
    parser.add_argument('--dataset', type=str, default='loop-ligand')
    parser.add_argument('--model_name', type=str, default='GCN_Trans')
    # 模型超参数
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--nhead', type=int, default=16)
    parser.add_argument('--transformer_encoder_layer', type=int, default=2)
    parser.add_argument('--num_base', type=int, default=6)
    parser.add_argument('--k_value', type=int, default=3)

    # 训练超参数
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    
    # 特征参数
    parser.add_argument('--dim_atom', type=int, default=78)
    parser.add_argument('--max_seq_len', type=int, default=24)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--val_size', type=float, default=0.1)

    # 输出参数
    parser.add_argument('--n_output', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--aux_weight', type=float, default=0.3)

    args = parser.parse_args()

    # 动态添加 device 和 time_str
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.time_str = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
    

    # 转成字典（如果你需要字典形式）
    args_dict = vars(args)
    return args_dict


if __name__ == '__main__':
    args = get_args()
    
    seed = args['seed']
    set_seed(seed)
    
    data_loaders = load_data(args)
    TraValTes(args, data_loaders)
    

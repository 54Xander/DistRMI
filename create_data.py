import copy
import itertools
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold

from utils import smile_to_graph
from utils import get_smiles_graph
from utils import load_site_distances
from utils import get_kmer_index
from utils import get_kmer_dict
from utils import TestbedDataset


def read_raw_data_loop_ligand(dataset, n_splits, seed, val_size ):
    print(f'convert data from DeepDTA for {dataset}')
    fpath = f'data/{dataset}/'

    # 读取配体和 RNA 序列数据
    df_ligands = pd.read_excel(fpath + "Ligand.xlsx")
    ligands = {row['Ligand_Data_ID']: row['CanonicalSMILES'] for index, row in df_ligands.iterrows()}
    
    df_rnas = pd.read_excel(fpath + "loop.xlsx")
    df_rnas.set_index('Loop_Data_ID', inplace=True)
    
    df_labels = pd.read_excel(fpath + "loop-ligand.xlsx")
    
    # 生成 n_splits 折交叉验证的索引
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_folds = []
    fold = 1
    for train_index, test_index in skf.split(df_labels, df_labels['Label']):
        print(f'Processing {dataset} fold {fold}')

        # 创建训练集和测试集
        df_tra = df_labels.iloc[train_index].copy()
        df_tes = df_labels.iloc[test_index].copy()

        # 从训练集中划分出20%作为验证集
        df_tra, df_val = train_test_split(df_tra, test_size=val_size, random_state=seed, stratify=df_tra['Label'])
    
        # 合并RNA序列数据
        df_tra = df_tra.merge(df_rnas, left_on='Loop_Data_ID', right_index=True)
        df_val = df_val.merge(df_rnas, left_on='Loop_Data_ID', right_index=True)
        df_tes = df_tes.merge(df_rnas, left_on='Loop_Data_ID', right_index=True)
    
        # 添加配体的SMILES信息
        df_tra['CanonicalSMILES'] = df_tra['Ligand_Data_ID'].map(ligands)
        df_val['CanonicalSMILES'] = df_val['Ligand_Data_ID'].map(ligands)
        df_tes['CanonicalSMILES'] = df_tes['Ligand_Data_ID'].map(ligands)
    
        # 只保留需要的列
        df_tra = df_tra[['Loop_Data_ID', '1D Sequence',
                         'Ligand_Data_ID', 'CanonicalSMILES', 'Label']]
        df_val = df_val[['Loop_Data_ID', '1D Sequence',
                         'Ligand_Data_ID', 'CanonicalSMILES', 'Label']]
        df_tes = df_tes[['Loop_Data_ID', '1D Sequence',
                         'Ligand_Data_ID', 'CanonicalSMILES', 'Label']]
    
        # 去除缺失值
        df_tra.dropna(subset=['CanonicalSMILES'], inplace=True)
        df_val.dropna(subset=['CanonicalSMILES'], inplace=True)
        df_tes.dropna(subset=['CanonicalSMILES'], inplace=True)
    
        os.makedirs('data/processed', exist_ok=True)
    
        # 保存为CSV文件
        df_tra.to_csv(f'data/processed/{dataset}_fold{fold}_tra.csv', index=True)
        df_val.to_csv(f'data/processed/{dataset}_fold{fold}_val.csv', index=True)
        df_tes.to_csv(f'data/processed/{dataset}_fold{fold}_tes.csv', index=True)
        
        # 将当前折的数据添加到列表中
        all_folds.append((df_tra, df_val, df_tes))
        
        fold += 1

    # 在所有折都处理完毕后返回数据集
    return all_folds



def trans(dataset, df_data, tvt_type, seed, k_value, max_seq_len, fold):
    drug_id = list(df_data['Ligand_Data_ID'])
    drug_smi = list(df_data['CanonicalSMILES'])
    rna_seq = list(df_data['1D Sequence'])
    rna_id = list(df_data['Loop_Data_ID'])
    Y = np.asarray(df_data['Label'])
    
    base_ids = []
    for ligand_id, loop_id, seq in zip(drug_id, rna_id, rna_seq):
        base_id = [(ligand_id, loop_id, i) for i in range(1, len(seq) + 1)]
        base_ids.append(base_id)

    site_dist_dict = load_site_distances()
    
    ligand_loop_dist = {}
    for ligand_id, loop_id, seq in zip(drug_id, rna_id, rna_seq):
        temp = []
        for i in range(1, max_seq_len + 1):
            dist = site_dist_dict.get((ligand_id, loop_id, i), float('inf'))
            temp.append(dist)
        ligand_loop_dist[(ligand_id, loop_id)] = np.array(temp)
    
    kmer_dict = get_kmer_dict(k_value)
    k_mer_features = [get_kmer_index(seq, kmer_dict, k_value, max_seq_len) for seq in rna_seq]
    
    smile_graph = get_smiles_graph(dataset)
    pyg_data = TestbedDataset(
        root='data',
        dataset=f'{dataset}_fold{fold}_{tvt_type}',
        xd=drug_smi,
        xt=rna_seq,
        y=Y,
        smile_graph=smile_graph,
        k_mer_features=k_mer_features,
        drug_id=drug_id,
        rna_id=rna_id,
        ligand_loop_dist=ligand_loop_dist
    )
    return pyg_data  # 返回处理后的数据集对象


if __name__ == '__main__':
    dataset = 'loop-ligand'
    n_splits = 5
    val_size= 0.1
    seed = 42
    k_value = 3
    max_seq_len = 24
    
    if dataset == 'loop-ligand':
        all_folds = read_raw_data_loop_ligand(dataset, n_splits, seed, val_size)

    for fold, (df_tra, df_val, df_tes) in enumerate(all_folds, start=1):
        trans(dataset, df_tra, 'tra', seed, k_value, max_seq_len, fold)
        trans(dataset, df_val, 'val', seed, k_value, max_seq_len, fold)
        trans(dataset, df_tes, 'tes', seed, k_value, max_seq_len, fold)
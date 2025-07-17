# utils.py

import torch
import numpy as np
from scipy import stats  # 导入 scipy.stats 模块
from torch_geometric.data import InMemoryDataset, Data
import os
from sklearn.metrics import matthews_corrcoef, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve, auc, mean_squared_error, confusion_matrix, average_precision_score
from rdkit import Chem
import networkx as nx
import itertools
import pandas as pd
import random

# 固定随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




# 定义辅助函数
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) + one_of_k_encoding(atom.GetDegree(),
                                                                                             [0, 1, 2, 3, 4, 5, 6, 7, 8,
                                                                                              9,
                                                                                              10]) + one_of_k_encoding_unk(
        atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + one_of_k_encoding_unk(atom.GetImplicitValence(),
                                                                                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                                           10]) + [
                        atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_smiles_graph(dataset):
    fpath = f'data/{dataset}/'
    # 读取配体和 RNA 序列数据
    df_ligands = pd.read_excel(fpath + "Ligand.xlsx")
    # 初始化两个空字典，用于存储配体的数据
    ligands = set(df_ligands['CanonicalSMILES'].values)
    smile_graph = {}
    for smile in ligands:
        try:
            g = smile_to_graph(smile)
            smile_graph[smile] = g
        except Exception as e:
            print(f"Error parsing SMILES: {smile} - {e}")
    return smile_graph


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index

def get_kmer_dict(k_value):
    # 定义字母集合
    bases = ['A', 'U', 'C', 'G', 'X', 'N']

    # 使用 itertools.product 生成所有可能的3-mer
    all_k_mers = [''.join(p) for p in itertools.product(bases, repeat=k_value)]

    # 创建字典，将每个3-mer 映射到一个索引
    kmer_dict = {kmer: idx+1 for idx, kmer in enumerate(all_k_mers)}
    return kmer_dict


def get_kmer_index(sequence, kmer_dict, k_value, max_seq_len):
    k_ls = []
    for i in range(len(sequence) - k_value + 1):
        kmer = sequence[i:i + k_value]
        index = kmer_dict[kmer]
        k_ls.append(index)
    while len(k_ls) < max_seq_len:
        k_ls.append(0)
    k_ls = k_ls[:max_seq_len]
    return np.array(k_ls, dtype=np.int64)  # 返回整数类型的 NumPy 数组



def masked_mean_pooling(x, mask):
    """
    x: [B, T, D] — 表示嵌入
    mask: [B, T] — True 表示 padding，需要忽略
    return: [B, D] — 平均池化后的表示
    """
    # 将 mask 反转：True 表示有效位置（非 padding）
    valid_mask = ~mask  # [B, T]，True表示可用
    # 为了广播乘法扩展维度：[B, T, 1]
    valid_mask = valid_mask.unsqueeze(-1)
    # 将 mask 应用到 x
    x_masked = x * valid_mask  # padding位置为0
    # 每个样本有多少有效 token（非 padding）
    valid_counts = valid_mask.sum(dim=1)  # [B, 1]
    # 避免除以0（全部是 padding）
    valid_counts = valid_counts.clamp(min=1)
    # 求平均
    pooled = x_masked.sum(dim=1) / valid_counts  # [B, D]
    return pooled

# 加载位点距离信息
def load_site_distances():
    file_path='data/site_distance.xlsx'
    distances_df = pd.read_excel(file_path)
    distance_dict = {}
    for _, row in distances_df.iterrows():
        ligand_id = row['Ligand_Data_ID']
        loop_id = row['Loop_Data_ID']
        base_id = row['New base ID']
        distance = row['Distance']
        if pd.isna(distance):
            distance = float('inf')  # 空的距离用无穷大表示
        distance_dict[(ligand_id, loop_id, base_id)] = distance
    return distance_dict

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='data', dataset='default_dataset',
                 xd=None, xt=None, y=None, k_mer_features=None, 
                 drug_id = None, rna_id = None, ligand_loop_dist=None, 
                 transform=None, pre_transform=None, smile_graph=None):
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.xd = xd
        self.xt = xt
        self.y = y
        self.k_mer_features = k_mer_features
        self.drug_id = drug_id
        self.rna_id = rna_id
        self.ligand_loop_dist = ligand_loop_dist
        self.smile_graph = smile_graph
        self.data, self.slices = self.load_or_process_data()

    def load_or_process_data(self):
        # 检查是否已经处理过数据
        if os.path.exists(self.processed_paths[0]):
            print(f"Loading processed data from {self.processed_paths[0]}")
            return torch.load(self.processed_paths[0])
        else:
            print("Processing data...")
            data_list = self.process(self.xd, self.xt, self.y, self.smile_graph, self.k_mer_features, self.ligand_loop_dist)
            torch.save(self.collate(data_list), self.processed_paths[0])
            return self.collate(data_list)

    def process(self, xd, xt, y, smile_graph, k_mer_features, ligand_loop_dist):
        drug_id = self.drug_id
        rna_id = self.rna_id
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        for i in range(len(xd)):
            smiles = xd[i]
            rna_sequence = xt[i] # RNA 序列
            label = y[i]
            ligand_id = drug_id[i]
            loop_id = rna_id[i]

            # 检查是否所有输入数据都存在
            if smiles not in smile_graph:
                raise ValueError(f"SMILES string '{smiles}' not found in smile_graph dictionary.")
            if k_mer_features is None or ligand_loop_dist is None:
                raise ValueError("k_mer_features or ligand_loop_dist is None. Please check the input data.")

            c_size, features, edge_index = smile_graph[smiles]

            # 使用 k-mer 特征
            k_mer_feature = k_mer_features[i]

            # Ensure edge_index is 2D
            edge_index = torch.LongTensor(edge_index).t() if edge_index else torch.LongTensor([[0], [0]])

            # Convert features and labels to tensors
            features = np.array(features, dtype=np.float32)
            label = np.array(label, dtype=np.int64)

            k_mer_feature = np.array(k_mer_feature, dtype=np.int64)
            dist = ligand_loop_dist[(ligand_id, loop_id)]

            # Create DATA.Data object
            data = Data(
                x=torch.from_numpy(features),
                edge_index=edge_index,
                y=torch.from_numpy(label),
                rna_sequence=torch.from_numpy(k_mer_feature),
                c_size=torch.tensor([c_size], dtype=torch.long),
                ligand_loop_dist=torch.FloatTensor(dist)
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        if not data_list:
            raise ValueError("No valid data points found. Check the input data.")

        return data_list

    @property
    def raw_file_names(self):
        return [f"{self.dataset}.csv"]  # 假设原始数据是一个 CSV 文件

    @property
    def processed_file_names(self):
        return [f"{self.dataset}.pt"]

    def download(self):
        pass  # 如果需要下载数据，可以在这里实现

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)



def get_metrics(real_score, predict_score):
    real_score = np.array(real_score)
    predict_score = np.array(predict_score)
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [round(auc[0, 0], 4), round(aupr[0, 0], 4), round(f1_score, 4),
            round(accuracy, 4), round(recall, 4), round(specificity, 4),
            round(precision, 4)]

if __name__ == '__main__':
    dataset = 'ligand' 
    data = TestbedDataset(root='data', dataset=dataset)


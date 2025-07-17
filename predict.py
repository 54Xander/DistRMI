import torch
import pandas as pd
from torch_geometric.data import Data, Batch
from utils import (
    get_kmer_index,
    get_kmer_dict,
    smile_to_graph
)
from models.gcn_transformer import GCN_Trans
import os
import argparse
import numpy as np
from rdkit import Chem

# 加载配置文件
def load_config():
    config = {
        'dataset': 'loop-ligand',
        'model_name': 'GCN_Trans',
        'embed_dim': 128,
        'nhead': 16,
        'transformer_encoder_layer': 2,
        'num_base': 6,
        'k_value': 3,
        'dim_atom': 78,
        'max_seq_len': 24,
        'n_output': 1,
        'alpha': 0.1,
        'aux_weight': 0.3,
        'dropout': 0.4,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    return config

# 预处理输入数据
def preprocess_input(smiles_list, rna_sequence_list, args):
    data_list = []
    kmer_dict = get_kmer_dict(args['k_value'])
    for smiles, rna_sequence in zip(smiles_list, rna_sequence_list):
        graph = smile_to_graph(smiles)
        c_size, features, edge_index = graph
        features_array = np.array(features)
        edge_index_array = np.array(edge_index)
        k_mer_feature = get_kmer_index(rna_sequence, kmer_dict, args['k_value'], args['max_seq_len'])

        x = torch.tensor(features_array, dtype=torch.float32)
        edge_index = torch.tensor(edge_index_array, dtype=torch.long).t().contiguous()
        rna_sequence_tensor = torch.tensor(k_mer_feature, dtype=torch.int64).unsqueeze(0)
        c_size_tensor = torch.tensor([c_size], dtype=torch.long)
        ligand_loop_dist = torch.zeros((1, args['max_seq_len']))

        data = Data(
            x=x,
            edge_index=edge_index,
            rna_sequence=rna_sequence_tensor,
            c_size=c_size_tensor,
            ligand_loop_dist=ligand_loop_dist,
            ligand_id="Ligand_Data_ID",
            loop_id="Loop_Data_ID",
            y=torch.tensor([0], dtype=torch.float)
        )
        data_list.append(data)

    batch = Batch.from_data_list(data_list)
    return batch

# 预测函数（仅输出预测结果）
def predict(model, data, device, loop_ids, ligand_ids):
    model.to(device)
    data.to(device)
    all_predictions = []

    for i in range(len(data)):
        sample_data = Batch.from_data_list([data[i]])
        with torch.no_grad():
            output = model(sample_data)
            pred_prob = torch.sigmoid(output['out']).item()
            pred = 1 if pred_prob >= 0.5 else 0

        loop_id = loop_ids[i] if loop_ids[i] is not None else f"loop_{i}"
        ligand_id = ligand_ids[i] if ligand_ids[i] is not None else f"ligand_{i}"

        all_predictions.append({
            'loop_id': loop_id,
            'ligand_id': ligand_id,
            'prediction': pred
        })

    return all_predictions

# 主程序
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RNA-Molecule Interaction Prediction')
    parser.add_argument('--model_path', type=str, default='json/GCN_Trans_loop-ligand_fold1.json',
                        help='Path to the trained model')
    parser.add_argument('--wdir', type=str, default='predict', help='Working directory for input and output files')
    args = parser.parse_args()

    config = load_config()
    model = GCN_Trans(config)
    model.load_state_dict(torch.load(args.model_path, map_location=config['device']))
    model.eval()

    predict_dir = args.wdir
    input_filename = 'Case.xlsx'
    input_path = os.path.join(predict_dir, input_filename)

    if not os.path.exists(input_path):
        print(f"文件 {input_path} 不存在，请确保文件已放入 '{predict_dir}' 文件夹中")
    else:
        df = pd.read_excel(input_path)
        valid_rows = df.dropna(subset=['RNA_loop_Sequence', 'SMILES'])
        rna_sequence_list = valid_rows['RNA_loop_Sequence'].tolist()
        smiles_list = valid_rows['SMILES'].tolist()
        loop_ids = valid_rows.get('Loop_Data_ID', [None] * len(valid_rows)).tolist()
        ligand_ids = valid_rows.get('Ligand_Data_ID', [None] * len(valid_rows)).tolist()
        RNA_Chain_ID = valid_rows.get('RNA Chain ID', [None] * len(valid_rows)).tolist()
        Ligand_Name = valid_rows.get('Ligand Name', [None] * len(valid_rows)).tolist()

        data_batch = preprocess_input(smiles_list, rna_sequence_list, config)
        predictions = predict(model, data_batch, torch.device(config['device']), loop_ids, ligand_ids)

        result_df = pd.DataFrame({
            'Loop_Data_ID': [p['loop_id'] for p in predictions],
            'Ligand_Data_ID': [p['ligand_id'] for p in predictions],
            'Prediction': [p['prediction'] for p in predictions],
            'RNA_Chain_ID': RNA_Chain_ID,
            'RNA_loop_Sequence': rna_sequence_list,
            'Ligand_Name': Ligand_Name,
            'SMILES': smiles_list
        })

        output_filename = f"predicted_{input_filename}"
        output_path = os.path.join(predict_dir, output_filename)
        result_df.to_excel(output_path, index=False)
        print(f"预测结果已保存到 {output_path}")
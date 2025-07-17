import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn.utils.rnn import pad_sequence
from utils import masked_mean_pooling
import os
from torch_geometric.nn import global_mean_pool as gap

os.environ["TORCH_NESTED"] = "0"
# 添加位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class MyMHA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        embed_dim = args['embed_dim']
        
        num_heads = args['nhead']
        
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.prior_weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_q, x_k, x_v, prior_attn=None, padding_mask=None):
        # x: [batch_size, seq_len, embed_dim]
        B, T_q, D = x_q.shape
        T_kv = x_k.shape[1]
        H = self.num_heads
        d_k = self.head_dim
        
        # 投影并reshape成多头格式
        Q = self.q_proj(x_q).view(B, T_q, H, d_k).transpose(1, 2)  # [B, H, T, d_k]
        K = self.k_proj(x_k).view(B, T_kv, H, d_k).transpose(1, 2)  # [B, H, T, d_k]
        V = self.v_proj(x_v).view(B, T_kv, H, d_k).transpose(1, 2)  # [B, H, T, d_k]

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # [B, H, T, T]

        # 添加 mask：把padding的位置设为一个很小的值
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            attn_scores = attn_scores.masked_fill(padding_mask, float('-inf'))

        # Softmax 得到 attention 权重
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, T, T]

        # 融合先验 attention 权重（你提供的）
        prior_weight = torch.sigmoid(self.prior_weight)
        if prior_attn is not None:
            # prior_attn: [B, T, T] -> broadcast到 [B, H, T, T]
            prior_attn = prior_attn.unsqueeze(1).expand(-1, H, -1, -1)
            attn_weights = (1 - prior_weight) * attn_weights + prior_weight * prior_attn

        # 乘上 V 得到 attention 输出
        attn_output = torch.matmul(attn_weights, V)  # [B, H, T, d_k]

        # 合并 heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, D)

        return self.out_proj(attn_output), attn_weights


class GCN_Trans(torch.nn.Module):
    def __init__(self, args):
        super(GCN_Trans, self).__init__()
        self.args = args
        dim_atom = args['dim_atom']
        dim_hid_drug = args['embed_dim']
        dropout = args['dropout']
        num_base = args['num_base']
        k_value = args['k_value']
        dim_emb_rna = args['embed_dim']
        nhead = args['nhead']
        transformer_encoder_layer = args['transformer_encoder_layer']
        self.max_seq_len = max_seq_len = args['max_seq_len']
        n_output = args['n_output']
        alpha = args['alpha']
        self.device = args['device']

        # SMILES graph branch (modified to two layers)
        self.drug_conv1 = GCNConv(dim_atom, dim_atom)
        self.drug_relu1 = nn.ReLU()
        self.drug_dropout1 = nn.Dropout(dropout)
        
        self.drug_conv2 = GCNConv(dim_atom, dim_atom * 2)
        self.drug_relu2 = nn.ReLU()
        self.drug_dropout2 = nn.Dropout(dropout)
        
        self.drug_fc_g1 = nn.Linear(dim_atom * 2, dim_hid_drug)  # Adjusted input dimension
        self.drug_relu3 = nn.ReLU()
        self.drug_dropout3 = nn.Dropout(dropout)
        
        self.drug_fc_g2 = nn.Linear(dim_hid_drug, dim_hid_drug)
        self.drug_dropout4 = nn.Dropout(dropout)

        # Transformer for RNA sequence (unchanged)
        self.embedding_xt = nn.Embedding(num_base ** k_value + 1, dim_emb_rna,
                                         padding_idx=0)
        self.pos_encoder = PositionalEncoding(dim_emb_rna, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_emb_rna, 
                                                   nhead=nhead, 
                                                   batch_first=True,
                                                   dropout=dropout,
                                                   dim_feedforward=dim_emb_rna)
        self.rna_transformer = nn.TransformerEncoder(encoder_layer, 
                                        num_layers=transformer_encoder_layer)
        self.rna_fc_1 = nn.Linear(dim_emb_rna, dim_emb_rna)
        
        self.MHA_drug_from_rna = MyMHA(args)
        self.MHA_rna_from_drug = MyMHA(args)
        
        self.dist_att = DistanceAwareAttention(dim_emb_rna, nhead, dropout, alpha)
        

        dim = int((dim_hid_drug + dim_emb_rna) / 2)
        # Combined layers (unchanged)
        self.fc1 = nn.Linear(dim_hid_drug * 2 + dim_emb_rna, dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(dim, dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.out = nn.Linear(dim, n_output)
        
        return

    def forward(self, data):
        drug_x, edge_index, batch = data.x, data.edge_index, data.batch
        c_size = data.c_size
        ligand_loop_dist = data.ligand_loop_dist
        

        # SMILES forward pass (modified to two layers)
        drug_x = self.drug_conv1(drug_x, edge_index)
        drug_x = self.drug_relu1(drug_x)
        drug_x = self.drug_dropout1(drug_x)
        
        drug_x = self.drug_conv2(drug_x, edge_index)
        drug_x = self.drug_relu2(drug_x)
        drug_x = self.drug_dropout2(drug_x)
        
        drug_x = self.drug_fc_g1(drug_x)
        drug_x = self.drug_relu3(drug_x)
        drug_x = self.drug_dropout3(drug_x)
        
        batch_drug = batch.max().item() + 1
        mol_atom_list = [drug_x[batch == i] for i in range(batch_drug)]
        drug_x_align = pad_sequence(mol_atom_list, batch_first=True)  # 有梯度，结构整洁
        drug_max_len = drug_x_align.shape[1]
        drug_mask = torch.zeros((batch_drug, drug_max_len), dtype=torch.bool)
        for i in range(batch_drug):
            num_atom = sum(batch == i).item()
            drug_mask[i, num_atom: ] = True # 之后都是padding
        drug_mask = drug_mask.to(self.device)
        
        drug_x_pool = gap(drug_x, batch)  # Global max pooling
        drug_x_pool = drug_x_pool.unsqueeze(1)
        drug_x_pool = self.drug_fc_g2(drug_x_pool)
        drug_x_pool = self.drug_dropout4(drug_x_pool)
        
        
        # Transformer encoder for RNA
        rna_sequence = data.rna_sequence
        rna_sequence = rna_sequence.view(-1, self.max_seq_len)
        rna_sequence = rna_sequence.long()
        
        x = self.embedding_xt(rna_sequence)              # (B, T, D)
        x = self.pos_encoder(x)        # 添加位置编码
        rna_mask = (rna_sequence == 0)  # mask: [batch_size, length]
        rna_output = self.rna_transformer(x, src_key_padding_mask=rna_mask)  # mask shape: [batch_size, seq_len]
        
        if rna_output.shape[1] < self.max_seq_len:
            batch_size, _, dim = rna_output.shape
            lack_time = self.max_seq_len - rna_output.shape[1]
            lack_x = torch.zeros((batch_size,lack_time,dim))
            lack_x = lack_x.to(self.device)
            rna_output = torch.cat((rna_output, lack_x), 1)
        rna_output = self.rna_fc_1(rna_output)
        
        # drug与RNA的交互
        drug_x, _ = self.MHA_drug_from_rna(drug_x_align, rna_output, rna_output, padding_mask=rna_mask)
        drug_x = masked_mean_pooling(drug_x, drug_mask)
        
        rna_x, _ = self.MHA_rna_from_drug(rna_output, drug_x_align, drug_x_align, padding_mask=drug_mask)
        rna_x = masked_mean_pooling(rna_x, rna_mask)
        
        distance_matrix = ligand_loop_dist.view(-1,self.max_seq_len)
        out_dist_aware, _, loss_aux = self.dist_att(drug_x_pool, rna_output, 
                                                    distance_matrix, rna_mask)

        # Concatenate drug and RNA features
        xc = torch.cat((drug_x, rna_x, out_dist_aware), 1)


        # Dense layers (unchanged)
        xc = self.fc1(xc)
        xc = self.relu1(xc)
        xc = self.dropout1(xc)
        
        xc = self.fc2(xc)
        xc = self.relu2(xc)
        xc = self.dropout2(xc)
        
        out = self.out(xc)
        output = {'out': out,
                  'loss_aux': loss_aux}
        return output


class DistanceAwareAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, alpha):
        super(DistanceAwareAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        # 可学习的融合参数 alpha，初值为 sigmoid⁻¹(alpha)
        init_val = math.log(alpha / (1 - alpha))  # inverse sigmoid for better init
        self.raw_alpha = nn.Parameter(torch.tensor(init_val))

    def forward(self, drug_feat, rna_feat, distance_matrix, mask=None):
        """
        drug_feat: Tensor [B, D]
        rna_feat: Tensor [B, T, D]
        distance_matrix: Tensor [B, T]
        mask: Optional bool Tensor [B, T] (True for PAD)
        """
        B, T, D = rna_feat.shape
        H = self.num_heads
        Dk = self.head_dim

        # Project Q, K, V
        q = self.q_proj(drug_feat).view(B, 1, H, Dk).transpose(1, 2)  # [B, H, 1, Dk]
        k = self.k_proj(rna_feat).view(B, T, H, Dk).transpose(1, 2)   # [B, H, T, Dk]
        v = self.v_proj(rna_feat).view(B, T, H, Dk).transpose(1, 2)   # [B, H, T, Dk]

        # Scaled dot-product attention
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, 1, T]
        attn_weights = F.softmax(attn_score, dim=-1)

        # Distance score
        dist_score = torch.exp(-distance_matrix.clamp(min=1e-6))        # [B, T]
        dist_score = dist_score.unsqueeze(1).unsqueeze(2).expand(-1, H, -1, -1)  # [B, H, 1, T]

        # Learnable fusion weight alpha ∈ (0, 1)
        alpha = torch.sigmoid(self.raw_alpha)

        # Fusion of scores
        if False:
            fused_weights = (1 - alpha) * attn_weights + alpha * dist_score
            loss_aux = None
        elif self.training:
            fused_weights = (1 - alpha) * attn_weights + alpha * dist_score
            loss_aux = nn.MSELoss()(attn_weights, dist_score)
        else:
            fused_weights = 1 * attn_weights
            loss_aux = None
        fused_weights = fused_weights / fused_weights.sum(dim=-1, keepdim=True)

        # Apply mask if given
        if mask is not None:
            fused_weights = fused_weights.masked_fill(mask.unsqueeze(1).unsqueeze(2), 0)

        fused_weights = self.dropout(fused_weights)
        out = torch.matmul(fused_weights, v)                            # [B, H, 1, Dk]
        out = out.transpose(1, 2).contiguous().view(B, 1, D)            # [B, 1, D]
        out = self.out_proj(out).squeeze(1)                             # [B, D]

        return out, fused_weights, loss_aux  # [B, D], [B, H, 1, T]

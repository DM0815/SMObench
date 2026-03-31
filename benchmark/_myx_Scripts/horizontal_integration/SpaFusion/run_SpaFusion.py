#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import time
import json
import torch
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ===========================
# Setup Paths
# ===========================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

# Add SpaFusion to path
spafusion_path = os.path.join(project_root, "Methods/SpaFusion")
sys.path.append(spafusion_path)

# ===========================
# Import SpaFusion modules
# ===========================
from main import (
    pre_train as _pre_train_orig,
    train as _train_orig,
    setup_seed,
    norm_adj,
    adjacent_matrix_preprocessing,
    load_data
)
from high_order_matrix import process_adjacency_matrix
from Utils.SMOBench_clustering import universal_clustering

# ---------------------------------------------------------------------------
# Monkey-patches for SpaFusion to fit Mouse_Thymus (17,824 cells) on A100-40GB.
#
# Root cause: SpaFusion creates 16 dense [n,n] matrices in forward pass:
#   - 4 encoder z_adj: sigmoid(z_x @ z_x.T)
#   - 4 decoder adj_hat: sigmoid(x_hat @ x_hat.T)
#   - 4 combined a_hat = z_adj + adj_hat
#   - 4 from emb_fusion: z_l @ z_l.T (attention-like)
# For n=17824, each float16 matrix = 635 MB → ~10 GB total.
#
# Fixes applied:
#   1. Skip dense adj reconstruction in encoder/decoder (→ set adj loss to 0)
#   2. Replace emb_fusion's dense attention with memory-efficient SDPA
#   3. AMP (mixed precision) + gradient checkpointing + sampled MSE
# ---------------------------------------------------------------------------
import torch.nn.functional as _F
import torch.optim as _optim
from encoder import GCNAutoencoder as _GCNAutoencoder, GCNEncoder as _GCNEncoder, GCNDecoder as _GCNDecoder
from utils import clustering as _clustering, distribution_loss as _distribution_loss, \
    target_distribution as _target_distribution, assignment as _assignment

# --- Patch 1: GCNEncoder — skip dense z_adj = sigmoid(z_x @ z_x.T) ---
def _enc_forward_no_adj(self, x, adj):
    x = self.layer1(x, adj, active=True)
    x = self.dropout(x)
    x = self.layer2(x, adj, active=True)
    x = self.dropout(x)
    z_x = self.layer3(x, adj, active=False)
    # Skip: z_adj = torch.sigmoid(torch.mm(z_x, z_x.t()))  # saves [n,n] matrix
    return z_x, None
# _GCNEncoder.forward = _enc_forward_no_adj  # DISABLED: use original algorithm

# --- Patch 2: GCNDecoder — skip dense adj_hat = sigmoid(x_hat @ x_hat.T) ---
def _dec_forward_no_adj(self, z_x, adj):
    x_hat = self.layer4(z_x, adj, active=True)
    x_hat = self.layer5(x_hat, adj, active=True)
    x_hat = self.layer6(x_hat, adj, active=True)
    # Skip: adj_hat = torch.sigmoid(torch.mm(x_hat, x_hat.t()))  # saves [n,n] matrix
    return x_hat, None
# _GCNDecoder.forward = _dec_forward_no_adj  # DISABLED: use original algorithm

# --- Patch 3: emb_fusion — replace dense attention with SDPA ---
def _emb_fusion_sdpa(self, adj, z_1, z_2, z_3):
    total = self.a + self.b + self.c
    z_i = (self.a / total) * z_1 + (self.b / total) * z_2 + (self.c / total) * z_3
    z_l = torch.spmm(adj, z_i)
    # Original: s = softmax(z_l @ z_l.T); z_g = s @ z_l  (creates [n,n] matrix)
    # Replacement: memory-efficient SDPA (no [n,n] materialization)
    z_g = _F.scaled_dot_product_attention(
        z_l.unsqueeze(0), z_l.unsqueeze(0), z_l.unsqueeze(0)
    ).squeeze(0)
    z_tilde = self.alpha * z_g + z_l
    return z_tilde
# _GCNAutoencoder.emb_fusion = _emb_fusion_sdpa  # DISABLED: use original algorithm

# --- Patch 4: GCNAutoencoder.forward — skip adj reconstruction, return zeros ---
import torch
_orig_autoencoder_forward = _GCNAutoencoder.forward

def _autoencoder_forward_no_adj(self, x1, adj1, adj2, x2, adj3, adj4, Mt1, Mt2, pretrain=False):
    adj2 = self.k1 * adj2 + self.k2 * Mt1
    adj4 = self.k1 * adj4 + self.k2 * Mt2

    z11, _ = self.encoder_view1(x1, adj1)
    z12, _ = self.encoder_view1(x1, adj2)
    z13 = self.trans_encoder1(x1.unsqueeze(0), mask=None).squeeze(0)

    z21, _ = self.encoder_view2(x2, adj3)
    z22, _ = self.encoder_view2(x2, adj4)
    z23 = self.trans_encoder2(x2.unsqueeze(0), mask=None).squeeze(0)

    z1_tilde = self.emb_fusion(adj2, z11, z12, z13)
    z2_tilde = self.emb_fusion(adj4, z21, z22, z23)
    z1_tilde = self.latent_process(z1_tilde)
    z2_tilde = self.latent_process(z2_tilde)

    w1 = torch.var(z1_tilde)
    w2 = torch.var(z2_tilde)
    a1 = w1 / (w1 + w2)
    Z = z1_tilde * a1 + z2_tilde * (1 - a1)

    # Decoder — feature reconstruction only (skip adj reconstruction)
    x11_hat, _ = self.decoder_view1(z11, adj1)
    x12_hat, _ = self.decoder_view1(z12, adj2)
    x13_hat = self.trans_decoder1(x=z1_tilde.unsqueeze(0), enc_out=z1_tilde.unsqueeze(0),
                                   src_mask=None, trg_mask=None).squeeze(0)

    x21_hat, _ = self.decoder_view2(z21, adj3)
    x22_hat, _ = self.decoder_view2(z22, adj4)
    x23_hat = self.trans_decoder2(x=z2_tilde.unsqueeze(0), enc_out=z2_tilde.unsqueeze(0),
                                   src_mask=None, trg_mask=None).squeeze(0)

    # Return scalar zeros for adj losses (they'll be multiplied by weight=0 in loss)
    _zero = torch.tensor(0.0, device=x1.device, requires_grad=True)

    Q = None if pretrain else self.q_distribution1(Z, z1_tilde, z2_tilde)
    return Z, z1_tilde, z2_tilde, _zero, _zero, _zero, _zero, x13_hat, x23_hat, Q

# _GCNAutoencoder.forward = _autoencoder_forward_no_adj  # DISABLED: use original algorithm
# print("[patch] Encoder/Decoder: skip dense n×n adj reconstruction (saves ~10 GB)")
# print("[patch] emb_fusion: using memory-efficient SDPA (saves ~2.5 GB)")


def _sampled_mse(a, b, sample_rows=4096):
    """Compute MSE loss on a random row subset to reduce peak memory during backward.
    For large n×n matrices, full MSE backward is O(n²) in memory.
    Sampling rows gives an unbiased gradient estimate with O(sample_rows×n) memory.
    """
    if a.dim() == 0:  # scalar (adj reconstruction was skipped)
        return a
    n = a.shape[0]
    if n <= sample_rows:
        return _F.mse_loss(a, b)
    idx = torch.randperm(n, device=a.device)[:sample_rows]
    return _F.mse_loss(a[idx], b[idx])


def pre_train(x1, x2, spatial_adj1, feature_adj1, spatial_adj2, feature_adj2,
              Mt1, Mt2, y, n_clusters, num_epoch, device, weight_list, lr, dataset_name):
    model = _GCNAutoencoder(
        input_dim1=x1.shape[1], input_dim2=x2.shape[1],
        enc_dim1=256, enc_dim2=128, dec_dim1=128, dec_dim2=256,
        latent_dim=20, dropout=0.1, num_layers=2, num_heads1=1, num_heads2=1,
        n_clusters=n_clusters, n_node=x1.shape[0])
    dataset_name = dataset_name.replace("/", "_")
    model.to(device)
    optimizer = _optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    from torch.utils.checkpoint import checkpoint as _ckpt

    pretrain_loss = []
    for epoch in range(num_epoch):
        with torch.cuda.amp.autocast():
            # Gradient checkpointing: recompute forward during backward to save memory
            Z, z1_tilde, z2_tilde, a11_hat, a12_hat, a21_hat, a22_hat, x13_hat, x23_hat, _ = \
                _ckpt(model, x1, spatial_adj1, feature_adj1, x2, spatial_adj2, feature_adj2, Mt1, Mt2, True, use_reentrant=False)
            # Sampled MSE for n×n adjacency matrices to reduce peak memory
            loss_ae1 = _sampled_mse(a11_hat, spatial_adj1)
            loss_ae2 = _sampled_mse(a12_hat, feature_adj1)
            loss_ae3 = _sampled_mse(a21_hat, spatial_adj2)
            loss_ae4 = _sampled_mse(a22_hat, feature_adj2)
            loss_x1 = _F.mse_loss(x13_hat, x1)
            loss_x2 = _F.mse_loss(x23_hat, x2)
            loss = (weight_list[0]*loss_ae1 + weight_list[1]*loss_ae2 +
                    weight_list[2]*loss_ae3 + weight_list[3]*loss_ae4 +
                    weight_list[4]*loss_x1 + weight_list[5]*loss_x2)
        pretrain_loss.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print("Epoch: {:.0f}/{:.0f} ,loss:{:.8f}".format(epoch + 1, num_epoch, loss))

    save_path = Path(f"./pretrain/{dataset_name}_pre_model.pkl")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return Z, z1_tilde, z2_tilde

def train(x1, x2, spatial_adj1, feature_adj1, spatial_adj2, feature_adj2,
          Mt1, Mt2, y, n_clusters, num_epoch, lambda1, device, seed, lambda2,
          weight_list, lr, num, spatial_K, adj_K, dataset_name):
    from torch.utils.checkpoint import checkpoint as _ckpt
    dataset_name = dataset_name.replace("/", "_")
    model = _GCNAutoencoder(
        input_dim1=x1.shape[1], input_dim2=x2.shape[1],
        enc_dim1=256, enc_dim2=128, dec_dim1=128, dec_dim2=256,
        latent_dim=20, dropout=0.1, num_layers=2, num_heads1=1, num_heads2=1,
        n_clusters=n_clusters, n_node=x1.shape[0])
    model.to(device)
    model.load_state_dict(torch.load(f'./pretrain/{dataset_name}_pre_model.pkl', map_location='cpu'))

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            Z, z1_tilde, z2_tilde, *_ = \
                model(x1, spatial_adj1, feature_adj1, x2, spatial_adj2, feature_adj2, Mt1, Mt2)
    centers1 = _clustering(Z, y, n_clusters=n_clusters)
    model.cluster_centers1.data = torch.tensor(centers1, dtype=torch.float32).to(device)

    optimizer = _optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    train_losses = []
    for epoch in range(num_epoch):
        with torch.cuda.amp.autocast():
            Z, z1_tilde, z2_tilde, a11_hat, a12_hat, a21_hat, a22_hat, x13_hat, x23_hat, Q = \
                _ckpt(model, x1, spatial_adj1, feature_adj1, x2, spatial_adj2, feature_adj2, Mt1, Mt2, False, use_reentrant=False)
            loss_ae1 = _sampled_mse(a11_hat, spatial_adj1)
            loss_ae2 = _sampled_mse(a12_hat, feature_adj1)
            loss_ae3 = _sampled_mse(a21_hat, spatial_adj2)
            loss_ae4 = _sampled_mse(a22_hat, feature_adj2)
            loss_x1 = _F.mse_loss(x13_hat, x1)
            loss_x2 = _F.mse_loss(x23_hat, x2)
            dense_loss1 = torch.mean((Z - z1_tilde) ** 2)
            dense_loss2 = torch.mean((Z - z2_tilde) ** 2)
            loss_rec = (weight_list[0]*loss_ae1 + weight_list[1]*loss_ae2 +
                        weight_list[2]*loss_ae3 + weight_list[3]*loss_ae4 +
                        weight_list[4]*loss_x1 + weight_list[5]*loss_x2)
            L_KL1 = _distribution_loss(Q, _target_distribution(Q[0].data))
            loss = loss_rec + lambda1 * L_KL1 + lambda2 * (dense_loss1 + dense_loss2)
        train_losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f"Epoch: {epoch+1}/{num_epoch} ,loss:{loss:.8f}")

        if y is not None:
            y_arr = np.array(y)
            if y_arr.dtype.kind not in {'i','u'}:
                unique_labels = {label: idx for idx, label in enumerate(np.unique(y_arr))}
                y_arr = np.array([unique_labels[label] for label in y_arr])
            acc, f1, nmi, ari, ami, vms, fms, y_pred = _assignment((Q[0]).data, y_arr)
        else:
            y_pred = torch.argmax(Q[0].data, dim=1).cpu().numpy()

    result_dir = './results'
    os.makedirs(result_dir, exist_ok=True)
    if y is not None:
        with open(os.path.join(result_dir, f'{dataset_name}_performance.csv'), 'a') as f:
            f.write(f"seed:{seed}, lambda1:{lambda1}, lambda2:{lambda2}, "
                    f"spatial_k:{spatial_K}, adj_k:{adj_K}, weight_list:{weight_list}, ")
            f.write(f"{acc:.4f},{f1:.4f},{nmi:.4f},{ari:.4f},{ami:.4f},{vms:.4f},{fms:.4f}\n")
    np.save(os.path.join(result_dir, f'{dataset_name}_{num}_pre_label.npy'), y_pred)
    np.save(os.path.join(result_dir, f'{dataset_name}_{num}_laten.npy'), Z.detach().cpu().numpy())
    return Z, z1_tilde, z2_tilde

# print("[patch] SpaFusion pre_train/train patched with AMP (mixed precision)")

# ---------------------------------------------------------------------------
# Monkey-patch: replace SelfAttention.forward with memory-efficient SDPA
# (scaled_dot_product_attention) to avoid materializing the full n×n
# attention matrix.  Saves ~1-2 GB on Mouse_Thymus (17824 cells).
# ---------------------------------------------------------------------------
from encoder import SelfAttention as _SelfAttention

_orig_sa_forward = _SelfAttention.forward

def _sdpa_forward(self, values, keys, query, mask):
    N = query.shape[0]
    value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

    # Reshape to [N, seq_len, heads, head_dim] then transpose to [N, heads, seq_len, head_dim]
    v = values.reshape(N, value_len, self.heads, self.head_dim).transpose(1, 2)
    k = keys.reshape(N, key_len, self.heads, self.head_dim).transpose(1, 2)
    q = query.reshape(N, query_len, self.heads, self.head_dim).transpose(1, 2)

    # Use memory-efficient attention (flash attention when available)
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0,
        scale=1.0 / (self.embed_size ** 0.5),
    )
    # out: [N, heads, query_len, head_dim] -> [N, query_len, embed_size]
    out = out.transpose(1, 2).reshape(N, query_len, self.embed_size)
    out = self.fc_out(out)
    return out

# _SelfAttention.forward = _sdpa_forward  # DISABLED: use original algorithm
# print("[patch] SelfAttention patched with scaled_dot_product_attention (memory-efficient)")

# ============================================================
# Helper: Dataset name parsing
# ============================================================
def parse_dataset_info(args):
    """Extract dataset_name and subset_name from fusion paths."""
    if hasattr(args, 'dataset') and args.dataset:
        return args.dataset, "fusion"

    # Auto-parse from RNA_path
    name_map = {
        "HLN_Fusion": "HLN",
        "HT_Fusion": "HT",
        "ME_S1_Fusion": "MISAR_S1",
        "ME_S2_Fusion": "MISAR_S2",
        "Mouse_Thymus_Fusion": "Mouse_Thymus",
        "Mouse_Spleen_Fusion": "Mouse_Spleen",
        "Mouse_Brain_Fusion": "Mouse_Brain",
    }
    for key, val in name_map.items():
        if key in args.RNA_path:
            return val, "fusion"

    return "Unknown", "fusion"


# ============================================================
# Main function
# ============================================================
def main(args):
    total_start_time = time.time()

    # --------------------- Device setup ---------------------
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Starting SpaFusion horizontal integration on device:", device)

    # --------------------- Random seed ---------------------
    setup_seed(args.seed)

    # --------------------- Load data ---------------------
    adata_rna = sc.read_h5ad(args.RNA_path)
    adata_rna.var_names_make_unique()

    if not args.ADT_path:
        raise ValueError("SpaFusion requires both RNA and ADT (Protein) data!")

    adata_protein = sc.read_h5ad(args.ADT_path)
    adata_protein.var_names_make_unique()

    # Use existing labels if available
    if 'Spatial_Label' in adata_rna.obs.columns:
        label = adata_rna.obs['Spatial_Label'].values
        n_clusters = len(np.unique(label))
    else:
        label = None
        n_clusters = args.cluster_nums

    # --------------------- Preprocessing & Graphs ---------------------
    adata_rna, adata_protein = load_data(
        adata_omics1=adata_rna,
        view1="RNA",
        adata_omics2=adata_protein,
        view2="Protein",
        n_neighbors=args.spatial_k,
        k=args.adj_k
    )

    # Feature matrices
    data1 = adata_rna.obsm['feat'].copy()
    data2 = adata_protein.obsm['feat'].copy()

    # Construct adjacency matrices (unique per cell count to avoid cache conflicts)
    adj_path = Path("./pre_adj") / f"{args.dataset}_{adata_rna.n_obs}cells"
    adj_path.mkdir(parents=True, exist_ok=True)

    adj = adjacent_matrix_preprocessing(adata_rna, adata_protein, adj_path)

    # Normalize adjacency matrices
    feature_adj1 = norm_adj(adj['adj_feature_omics1'])
    feature_adj2 = norm_adj(adj['adj_feature_omics2'])
    spatial_adj1 = norm_adj(adj['adj_spatial_omics1'])
    spatial_adj2 = norm_adj(adj['adj_spatial_omics2'])

    # High-order adjacency matrices
    Mt1 = norm_adj(process_adjacency_matrix(feature_adj1, adj_path / f"{args.dataset}_Mt1.npy"))
    Mt2 = norm_adj(process_adjacency_matrix(feature_adj2, adj_path / f"{args.dataset}_Mt2.npy"))

    # Convert to torch tensors — float32 consistent with original SpaFusion code
    data1 = torch.tensor(data1, dtype=torch.float32).to(device)
    data2 = torch.tensor(data2, dtype=torch.float32).to(device)
    feature_adj1 = torch.tensor(feature_adj1, dtype=torch.float32).to(device)
    feature_adj2 = torch.tensor(feature_adj2, dtype=torch.float32).to(device)
    spatial_adj1 = torch.tensor(spatial_adj1, dtype=torch.float32).to(device)
    spatial_adj2 = torch.tensor(spatial_adj2, dtype=torch.float32).to(device)
    Mt1 = torch.tensor(Mt1, dtype=torch.float32).to(device)
    Mt2 = torch.tensor(Mt2, dtype=torch.float32).to(device)

    print("Adjacency tensor shape:", spatial_adj2.shape)

    # --------------------- Pretraining ---------------------
    print("\n========== Pretraining SpaFusion ==========")
    start_time_pretrain = time.time()

    emb_combination, emb_RNA, emb_ADT = _pre_train_orig(
        x1=data1,
        x2=data2,
        spatial_adj1=spatial_adj1,
        feature_adj1=feature_adj1,
        spatial_adj2=spatial_adj2,
        feature_adj2=feature_adj2,
        Mt1=Mt1,
        Mt2=Mt2,
        y=label,
        n_clusters=n_clusters,
        num_epoch=args.pretrain_epoch,
        device=device,
        weight_list=args.weight_list,
        lr=args.lr,
        dataset_name=args.dataset
    )

    pretrain_time = time.time() - start_time_pretrain
    print("SpaFusion Pretrain time:", pretrain_time)

    # --------------------- Training ---------------------
    print("\n========== Training SpaFusion ==========")
    start_time_train = time.time()

    for i in range(1):
        print(f"Training round {i}")
        emb_combination, emb_RNA, emb_ADT = train(
            x1=data1,
            x2=data2,
            spatial_adj1=spatial_adj1,
            feature_adj1=feature_adj1,
            spatial_adj2=spatial_adj2,
            feature_adj2=feature_adj2,
            y=label,
            n_clusters=n_clusters,
            Mt1=Mt1,
            Mt2=Mt2,
            num_epoch=args.train_epoch,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            device=device,
            seed=args.seed,
            weight_list=args.weight_list,
            lr=args.lr,
            num=i,
            spatial_K=args.spatial_k,
            adj_K=args.adj_k,
            dataset_name=args.dataset
        )

    train_time = time.time() - start_time_train
    print("SpaFusion Train time:", train_time)

    # --------------------- Build result AnnData ---------------------
    print("\n========== Building result AnnData ==========")
    adata = adata_rna.copy()
    adata.obsm['SpaFusion'] = emb_combination.detach().cpu().numpy()
    adata.obsm['emb_latent_omics1'] = emb_RNA.detach().cpu().numpy()
    adata.obsm['emb_latent_omics2'] = emb_ADT.detach().cpu().numpy()
    adata.uns.update({
        'train_time': train_time,
        'pretrain_time': pretrain_time,
        'integration_type': 'horizontal'
    })

    # --------------------- UMAP ---------------------
    print("\nGenerating UMAP coordinates...")
    sc.pp.neighbors(adata, use_rep='SpaFusion', n_neighbors=30)
    sc.tl.umap(adata)
    print("UMAP coordinates stored in adata.obsm['X_umap']")

    # --------------------- Clustering ---------------------
    print("\nRunning clustering methods...")
    tools = ['mclust', 'louvain', 'leiden', 'kmeans']
    for tool in tools:
        print(f"Running {tool} clustering...")
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='SpaFusion',
            method=tool,
            key=tool,
            use_pca=False
        )
    print("All clustering methods completed.")

    # --------------------- Save results ---------------------
    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir, exist_ok=True)
    adata.write(args.save_path)

    print(adata)
    print("SpaFusion horizontal integration results saved to:", args.save_path)

    # === Save Timing JSON ===
    total_time = time.time() - total_start_time
    dataset_name, _ = parse_dataset_info(args)
    timing_info = {
        "method": "SpaFusion",
        "dataset": dataset_name,
        "integration_type": "horizontal",
        "modality": "RNA_ADT",
        "n_cells": adata.n_obs,
        "embedding_shape": list(adata.obsm["SpaFusion"].shape),
        "pretrain_time_s": round(pretrain_time, 2),
        "training_time_s": round(train_time, 2),
        "total_time_s": round(total_time, 2),
        "device": str(device),
        "seed": args.seed,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    timing_path = args.save_path.replace(".h5ad", "_timing_info.json")
    with open(timing_path, "w") as f:
        json.dump(timing_info, f, indent=2)
    print(f"Timing info saved to {timing_path}")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    # Environment variables for R & threading
    os.environ['R_HOME'] = '/home/users/nus/e1724738/miniconda3/envs/_Proj1_1/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    print("Starting SpaFusion horizontal integration...")

    parser = argparse.ArgumentParser(description="Run SpaFusion horizontal integration for RNA+ADT")
    parser.add_argument('--RNA_path', type=str, required=True, help='Path to RNA AnnData (.h5ad)')
    parser.add_argument('--ADT_path', type=str, required=True, help='Path to ADT/Protein AnnData (.h5ad)')
    parser.add_argument('--dataset', type=str, default='D1', help='Dataset name')
    parser.add_argument('--save_path', type=str, required=True, help='Output path to save integrated AnnData')

    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device, e.g., cuda:0 or cpu')

    parser.add_argument('--spatial_k', type=int, default=9, help='Number of spatial neighbors')
    parser.add_argument('--adj_k', type=int, default=20, help='Number of adjacency neighbors')

    parser.add_argument('--pretrain_epoch', type=int, default=100, help='Number of pretraining epochs')
    parser.add_argument('--train_epoch', type=int, default=100, help='Number of training epochs')

    parser.add_argument('--lambda1', type=float, default=1.0, help='Lambda1 for clustering loss')
    parser.add_argument('--lambda2', type=float, default=0.1, help='Lambda2 for dense loss')
    parser.add_argument('--weight_list', type=list, default=[1, 1, 1, 1, 1, 1], help='Weights for reconstruction loss')

    parser.add_argument('--cluster_nums', type=int, default=None, help='Number of clusters for clustering')
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for SpaFusion training")

    args = parser.parse_args()
    main(args)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import scanpy as sc
import numpy as np
import argparse
import time
import json
import sys
import re
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

# Add project root directory to module search path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

# Add CANDIES to path
candies_path = os.path.join(project_root, "Methods/CANDIES/CANDIES/codes")
sys.path.append(candies_path)

try:
    from DiTs import *
    from sampler import *
    from train_diff import *
    from ZINB_encoder import *
    from preprocess1 import *
    from get_graph import construct_neighbor_graph, adjacent_matrix_preprocessing
    from modality_selection import modality_selection
    from integration import *
    from AutoEncoder import train_model, train_atac
except ImportError as e:
    print(f"Error importing CANDIES modules: {e}")
    print("Please ensure CANDIES is properly installed and accessible")
    sys.exit(1)

from Utils.SMOBench_clustering import universal_clustering


# =============================================================================
# MONKEY-PATCHES for large dataset OOM / NaN / timeout fixes
# These patches modify behavior at runtime WITHOUT touching Methods/ source code.
# =============================================================================

# --- Threshold for "large dataset" ---
_LARGE_DATASET_THRESHOLD = float('inf')  # Disabled: use original algorithm at all sizes

# --- Fix 1: Sampled NTXentLoss to avoid [2N, 2N] OOM ---
# For N>10k the original NTXentLoss creates a 2N x 2N similarity matrix that
# causes OOM. This replacement samples a fixed number of negatives per anchor.

class SampledNTXentLoss(nn.Module):
    """Memory-efficient NTXent loss that samples negatives instead of using all 2N-2."""

    def __init__(self, temperature: float = 0.07, max_negatives: int = 1024):
        super().__init__()
        self.temperature = temperature
        self.max_negatives = max_negatives
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z1, z2):
        batch_size = z1.shape[0]
        # If small enough, fall back to full computation
        if batch_size * 2 <= self.max_negatives + 2:
            return self._full_forward(z1, z2)
        return self._sampled_forward(z1, z2)

    def _full_forward(self, z1, z2):
        """Original full NTXent (for small N)."""
        batch_size = z1.shape[0]
        labels = torch.cat([torch.arange(batch_size)] * 2, dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z1.device)
        features = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z1.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        target = torch.zeros(logits.shape[0], dtype=torch.long).to(z1.device)
        return self.criterion(logits, target)

    def _sampled_forward(self, z1, z2):
        """Sampled NTXent: for each anchor, use its positive + K random negatives."""
        batch_size = z1.shape[0]
        features = F.normalize(torch.cat([z1, z2], dim=0), dim=1)  # (2N, D)
        total = 2 * batch_size
        K = min(self.max_negatives, total - 2)

        loss = torch.tensor(0.0, device=z1.device)
        # Process anchors in chunks to keep memory bounded
        chunk_size = min(512, total)
        n_chunks = (total + chunk_size - 1) // chunk_size

        for c in range(n_chunks):
            start = c * chunk_size
            end = min(start + chunk_size, total)
            anchors = features[start:end]  # (chunk, D)
            chunk_len = end - start

            # Positive index for anchor i: if i < N then i+N, else i-N
            pos_indices = torch.arange(start, end, device=z1.device)
            pos_indices = torch.where(pos_indices < batch_size,
                                      pos_indices + batch_size,
                                      pos_indices - batch_size)
            positives = features[pos_indices]  # (chunk, D)
            pos_sim = (anchors * positives).sum(dim=1, keepdim=True) / self.temperature  # (chunk, 1)

            # Sample K negative indices for this chunk (shared across anchors in chunk for speed)
            # Exclude self and positive -- approximate: sample from all, very unlikely to hit self/pos
            neg_indices = torch.randint(0, total, (K,), device=z1.device)
            neg_features = features[neg_indices]  # (K, D)
            neg_sim = torch.matmul(anchors, neg_features.T) / self.temperature  # (chunk, K)

            logits = torch.cat([pos_sim, neg_sim], dim=1)  # (chunk, 1+K)
            target = torch.zeros(chunk_len, dtype=torch.long, device=z1.device)
            loss = loss + self.criterion(logits, target) * chunk_len

        return loss / total


def _patched_train_and_infer(features_omics1, features_omics2,
                              adj_spatial_omics1, adj_feature_omics1,
                              adj_spatial_omics2, adj_feature_omics2,
                              epochs, device,
                              dim_out_feat_omics1=64, dim_out_feat_omics2=64,
                              learning_rate=0.001, weight_decay=0.00,
                              patience=10, min_delta=0.001):
    """
    Patched train_and_infer that uses SampledNTXentLoss for large datasets
    to avoid OOM from the [2N, 2N] similarity matrix.
    """
    from tqdm import tqdm as _tqdm

    N = features_omics1.shape[0]
    is_large = N > _LARGE_DATASET_THRESHOLD

    model = Encode_all(
        dim_in_feat_omics1=features_omics1.shape[1],
        dim_out_feat_omics1=dim_out_feat_omics1,
        dim_in_feat_omics2=features_omics2.shape[1],
        dim_out_feat_omics2=dim_out_feat_omics2
    )

    optimizer = torch.optim.Adam(model.parameters(), learning_rate,
                                 weight_decay=weight_decay)

    model = model.to(device)
    features_omics1 = features_omics1.to(device)
    features_omics2 = features_omics2.to(device)
    adj_spatial_omics1 = adj_spatial_omics1.to(device)
    adj_feature_omics1 = adj_feature_omics1.to(device)
    adj_spatial_omics2 = adj_spatial_omics2.to(device)
    adj_feature_omics2 = adj_feature_omics2.to(device)

    if is_large:
        print(f"[PATCH] N={N} > {_LARGE_DATASET_THRESHOLD}: using SampledNTXentLoss (max_negatives=1024)")
        contrastive_loss_fn = SampledNTXentLoss(temperature=0.07, max_negatives=1024).to(device)
    else:
        contrastive_loss_fn = model.NTXentLoss()

    best_loss = float('inf')
    patience_counter = 0

    model.train()
    with _tqdm(total=epochs, desc="Training Progress") as pbar:
        for epoch in range(epochs):
            model.train()
            results = model(features_omics1, features_omics2, adj_spatial_omics1, adj_feature_omics1,
                            adj_spatial_omics2, adj_feature_omics2)

            emb_latent_spatial_omics1 = results['emb_latent_spatial_omics1']
            emb_latent_spatial_omics2 = results['emb_latent_spatial_omics2']
            emb_latent_feature_omics1 = results['emb_latent_feature_omics1']
            emb_latent_feature_omics2 = results['emb_latent_feature_omics2']

            emb_recon_spatial_omics1 = results['emb_recon_spatial_omics1']
            emb_recon_spatial_omics2 = results['emb_recon_spatial_omics2']
            emb_recon_feature_omics1 = results['emb_recon_feature_omics1']
            emb_recon_feature_omics2 = results['emb_recon_feature_omics2']

            # Reconstruction losses
            recon_loss_spatial_omics1 = F.mse_loss(emb_recon_spatial_omics1, features_omics1)
            recon_loss_spatial_omics2 = F.mse_loss(emb_recon_spatial_omics2, features_omics2)
            recon_loss_feature_omics1 = F.mse_loss(emb_recon_feature_omics1, features_omics1)
            recon_loss_feature_omics2 = F.mse_loss(emb_recon_feature_omics2, features_omics2)

            # Contrastive losses
            contrastive_loss_spatial = contrastive_loss_fn(emb_latent_spatial_omics1, emb_latent_spatial_omics2)
            contrastive_loss_feature = contrastive_loss_fn(emb_latent_feature_omics1, emb_latent_feature_omics2)

            total_recon_loss = (recon_loss_spatial_omics1 + recon_loss_spatial_omics2 +
                                recon_loss_feature_omics1 + recon_loss_feature_omics2)
            total_contra_loss = contrastive_loss_spatial + contrastive_loss_feature
            total_loss = total_recon_loss + total_contra_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if total_loss.item() + min_delta < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered at epoch:", epoch + 1)
                    break

            pbar.set_postfix({
                "Total": f"{total_loss.item():.4f}",
                "Contra": f"{total_contra_loss.item():.4f}",
                "Recon": f"{total_recon_loss.item():.4f}",
            })
            pbar.update(1)

    model.eval()
    with torch.no_grad():
        final_results = model(features_omics1, features_omics2, adj_spatial_omics1, adj_feature_omics1,
                              adj_spatial_omics2, adj_feature_omics2)
    return final_results


# --- Fix 2: Monkey-patch normal_train_diff to use CosineAnnealingLR ---
# The original StepLR(step_size=100, gamma=0.1) kills LR too aggressively,
# causing NaN in the diffusion denoise phase for large datasets.

import train_diff as _train_diff_module

_original_normal_train_diff = _train_diff_module.normal_train_diff


def _patched_normal_train_diff(model, dataloader, lr=1e-4, num_epoch=1400,
                                pred_type='noise', diffusion_step=1000,
                                device=torch.device('cuda:0'), is_tqdm=True,
                                patience=20):
    """Patched normal_train_diff with CosineAnnealingLR instead of aggressive StepLR."""
    from diff_scheduler import NoiseScheduler
    from tqdm import tqdm as _tqdm

    noise_scheduler = NoiseScheduler(num_timesteps=diffusion_step, beta_schedule='cosine')
    criterion = _train_diff_module.diffusion_loss()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    # FIX: Use CosineAnnealingLR instead of StepLR(step_size=100, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr * 0.01)

    best_loss = float('inf')
    patience_counter = 0

    if is_tqdm:
        t_epoch = _tqdm(range(num_epoch), ncols=100)
    else:
        t_epoch = range(num_epoch)

    model.train()
    for epoch in t_epoch:
        epoch_loss = 0.0
        for i, (x, x_hat, x_cond) in enumerate(dataloader):
            x, x_hat, x_cond = x.float().to(device), x_hat.float().to(device), x_cond.float().to(device)
            x_noise = torch.randn(x.shape).to(device)
            timesteps = torch.randint(1, diffusion_step, (x.shape[0],)).to(device)
            x_t = noise_scheduler.add_noise(x, x_noise, timesteps=timesteps)
            x_noisy = x_t
            noise_pred = model(x_noisy, x_hat, t=timesteps.to(device), y=x_cond)
            loss = criterion(x_noise, noise_pred)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        scheduler.step()
        epoch_loss = epoch_loss / (i + 1)
        current_lr = optimizer.param_groups[0]['lr']

        if is_tqdm:
            t_epoch.set_postfix_str(f'{pred_type} loss:{epoch_loss:.7f}, lr:{current_lr:.2e}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stop(patience:{patience})!!")
                break

# Apply the monkey-patch
_train_diff_module.normal_train_diff = _patched_normal_train_diff
print("[PATCH] normal_train_diff monkey-patched: CosineAnnealingLR replaces StepLR(gamma=0.1)")


# --- Fix 3: Memory-efficient ConditionalDiffusionDataset ---
# The original returns self.con_data (full N x D matrix) per sample, causing
# DataLoader to create [batch_size, N, D] tensors. For N=37885 this is huge.
# Since DiT_diff.forward passes y through SimpleMLP which does x.mean(dim=1),
# we precompute the mean and return a 1-row "matrix" instead.

class EfficientConditionalDiffusionDataset:
    """Memory-efficient dataset that precomputes con_data mean."""
    def __init__(self, adata_omics1, adata_omics2):
        self.adata_omics1 = adata_omics1
        self.adata_omics2 = adata_omics2
        self.st_sample = torch.tensor(adata_omics1, dtype=torch.float32)
        self.con_sample = torch.tensor(adata_omics2, dtype=torch.float32)
        # Precompute mean: shape (1, D) so SimpleMLP's x.mean(dim=1) still works
        self.con_data = torch.tensor(adata_omics2, dtype=torch.float32).mean(dim=0, keepdim=True)
        print(f"[PATCH] EfficientConditionalDiffusionDataset: con_data shape {self.con_data.shape} "
              f"(precomputed mean, was ({adata_omics2.shape[0]}, {adata_omics2.shape[1]}))")

    def __len__(self):
        return len(self.st_sample)

    def __getitem__(self, idx):
        return self.st_sample[idx], self.con_sample[idx], self.con_data


# =============================================================================
# END MONKEY-PATCHES
# =============================================================================


def run_leiden_candies(adata1, n_cluster, use_rep="embeddings", key_added="Nleiden", range_min=0, range_max=3, max_steps=30, tolerance=0):
    """CANDIES style leiden clustering with resolution search"""
    adata = adata1.copy()
    sc.pp.neighbors(adata, use_rep=use_rep)
    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    
    while this_step < max_steps:
        this_resolution = this_min + ((this_max-this_min)/2)
        sc.tl.leiden(adata, resolution=this_resolution)
        this_clusters = adata.obs['leiden'].nunique()

        if this_clusters > n_cluster+tolerance:
            this_max = this_resolution
        elif this_clusters < n_cluster-tolerance:
            this_min = this_resolution
        else:
            print("Succeed to find %d clusters at resolution %.3f"%(n_cluster, this_resolution))
            adata1.obs[key_added] = adata.obs["leiden"]
            return adata1
        
        this_step += 1
    
    print('Cannot find the number of clusters')
    adata1.obs[key_added] = adata.obs["leiden"]
    return adata1


def clustering_candies(adata, key='embeddings', add_key='clusters', n_clusters=10, end=2, method='mclust', use_pca=True):
    """CANDIES style clustering function supporting mclust and leiden"""
    if method == 'mclust':
        from Utils.SMOBench_clustering import universal_clustering
        adata = universal_clustering(adata, n_clusters=n_clusters, used_obsm=key, 
                                   method='mclust', key=add_key, use_pca=use_pca)
    else:  # leiden or louvain
        adata = run_leiden_candies(adata, n_clusters, use_rep=key, key_added=add_key, range_max=end)
    
    return adata


def parse_dataset_info(args):
    """Extract dataset_name and subset_name from fusion paths"""
    if hasattr(args, 'dataset') and args.dataset:
        return args.dataset, "fusion"
    
    # Auto parse from RNA_path
    if "HLN_Fusion" in args.RNA_path:
        return "HLN", "fusion"
    elif "HT_Fusion" in args.RNA_path:
        return "HT", "fusion" 
    elif "ME_S1_Fusion" in args.RNA_path:
        return "MISAR_S1", "fusion"
    elif "ME_S2_Fusion" in args.RNA_path:
        return "MISAR_S2", "fusion"
    elif "Mouse_Thymus_Fusion" in args.RNA_path:
        return "Mouse_Thymus", "fusion"
    elif "Mouse_Spleen_Fusion" in args.RNA_path:
        return "Mouse_Spleen", "fusion"
    elif "Mouse_Brain_Fusion" in args.RNA_path:
        return "Mouse_Brain", "fusion"
    
    return "Unknown", "fusion"


def seed_everything(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    total_start_time = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # Set seeds
    seed_everything(args.seed)
    
    # === Load Fusion Data ===
    adata_omics1 = sc.read_h5ad(args.RNA_path)  # RNA fusion data
    adata_omics1.var_names_make_unique()

    if args.ADT_path:
        adata_omics2 = sc.read_h5ad(args.ADT_path)  # ADT fusion data
        modality = 'ADT'
        modality_name = 'Proteome'
        adata_omics2.var_names_make_unique()
    elif args.ATAC_path:
        adata_omics2 = sc.read_h5ad(args.ATAC_path)  # ATAC fusion data
        modality = 'ATAC'
        modality_name = 'Epigenome'
        adata_omics2.var_names_make_unique()
    else:
        raise ValueError("Either ADT_path or ATAC_path must be provided.")

    print(f"Processing horizontal integration: RNA + {modality_name} fusion data...")

    # === Ensure both datasets have same cells ===
    common_obs = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
    print(f"Common cells: {len(common_obs)}, RNA cells: {adata_omics1.n_obs}, {modality_name} cells: {adata_omics2.n_obs}")
    adata_omics1 = adata_omics1[common_obs].copy()
    adata_omics2 = adata_omics2[common_obs].copy()
    print(f"After intersection - RNA: {adata_omics1.n_obs}, {modality_name}: {adata_omics2.n_obs}")

    # Check batch information for horizontal integration
    if 'batch' in adata_omics1.obs.columns:
        print(f"Batch distribution in RNA: {adata_omics1.obs['batch'].value_counts()}")
        print("Horizontal integration will address batch effects between these batches")
    else:
        print("Warning: No 'batch' column found in fusion data. Adding default batch labels.")
        # Add default batch information if not present
        n_cells = adata_omics1.n_obs
        adata_omics1.obs['batch'] = ['batch_1'] * (n_cells // 2) + ['batch_2'] * (n_cells - n_cells // 2)
        adata_omics2.obs['batch'] = adata_omics1.obs['batch'].copy()

    # Add spatial coordinates for horizontal integration if not present
    if 'spatial' not in adata_omics1.obsm.keys():
        print("Warning: No spatial coordinates found. Generating pseudo-spatial coordinates for horizontal integration...")
        # Generate pseudo-spatial coordinates based on UMAP or PCA
        sc.pp.neighbors(adata_omics1)
        sc.tl.umap(adata_omics1)
        adata_omics1.obsm['spatial'] = adata_omics1.obsm['X_umap'].copy()
    
    if 'spatial' not in adata_omics2.obsm.keys():
        print("Generating pseudo-spatial coordinates for modality 2...")
        sc.pp.neighbors(adata_omics2)
        sc.tl.umap(adata_omics2)
        adata_omics2.obsm['spatial'] = adata_omics2.obsm['X_umap'].copy()

    # === Data preprocessing ===
    print("Data preprocessing for horizontal integration...")
    
    # RNA preprocessing
    sc.pp.filter_genes(adata_omics1, min_cells=10)
    sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1)

    if modality == 'ADT':
        # Protein preprocessing for horizontal integration
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars-1)
        
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars-1)
        
    else:  # ATAC preprocessing for horizontal integration
        if 'X_lsi' not in adata_omics2.obsm.keys():
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
            lsi(adata_omics2, use_highly_variable=False, n_components=51)
        
        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.obsm['X_lsi'].shape[1])

    # === Start training time recording ===
    start_time = time.time()
    
    # === Encoding phase ===
    print("Encoding phase for horizontal integration...")
    
    # Construct neighbor graphs (batch-aware for horizontal integration)
    adata_omics1, adata_omics2 = construct_neighbor_graph(adata_omics1, adata_omics2)
    adj = adjacent_matrix_preprocessing(adata_omics1, adata_omics2)
    adj_spatial_omics1 = adj['adj_spatial_omics1'].to(device)
    adj_spatial_omics2 = adj['adj_spatial_omics2'].to(device)

    # RNA encoding with ZINB (adapted for horizontal integration)
    ae_model = encoder_ZINB(
        adata=adata_omics1,
        device=device,
        epochs=300, 
        dim_output=64 if modality == 'ADT' else 128
    )
    adata_omics1.obsm['emb_ZINB'], adj_mat = ae_model.train()

    # Second modality encoding (horizontal integration specific)
    seed_everything(args.seed)
    if modality == 'ADT':
        train_model(adata_omics1, adata_omics2, adj_spatial_omics1, adj_spatial_omics2, epochs=400)
    else:  # ATAC
        train_atac(adata_omics2, adj_spatial_omics2, epochs=600)

    # === Batch-aware clustering for horizontal integration ===
    print("Batch-aware clustering for horizontal integration...")
    if modality == 'ADT':
        cluster_num = min(10, args.cluster_nums)
        adata_omics1 = run_leiden_candies(adata_omics1, n_cluster=cluster_num, use_rep="emb_ZINB", key_added="AE")
        adata_omics2 = run_leiden_candies(adata_omics2, n_cluster=cluster_num, use_rep="emb_latent_omics2", key_added="AE")
    else:  # RNA+ATAC
        cluster_num = min(14, args.cluster_nums)
        adata_omics1 = run_leiden_candies(adata_omics1, n_cluster=cluster_num, use_rep="emb_ZINB", key_added="AE")
        adata_omics2 = clustering_candies(adata_omics2, key='emb_latent_omics2', add_key='AE', n_clusters=cluster_num, method='mclust')

    # === Modality selection (adapted for horizontal integration) ===
    print("Modality selection for horizontal integration...")
    try:
        modality_selection(adata_omics1.obs['AE'], adata_omics2.obs['AE'], 
                         modality1_embs=adata_omics1.obsm['emb_ZINB'], 
                         modality2_embs=adata_omics2.obsm['emb_latent_omics2'], 
                         spatial_coor=adata_omics1.obsm['spatial'])
    except:
        print("Modality selection completed (using default for horizontal integration)")

    # === Denoise phase (horizontal integration specific) ===
    print("Denoise phase for horizontal integration...")
    
    # For horizontal integration, handle batch alignment properly
    slices_omics1_spatial = adata_omics1.obsm['spatial']
    slices_omics2_spatial = adata_omics2.obsm['spatial']
    emb_latent_omics1 = adata_omics1.obsm['emb_ZINB']
    
    if modality == 'ADT':
        # For ADT: create label embeddings from clusters (batch-aware)
        ad2_ae = adata_omics2.obs['AE']
        labels = torch.tensor(ad2_ae.values.astype(int), dtype=torch.long)
        num_classes = labels.max().item() + 1 
        embedding_layer = torch.nn.Embedding(num_classes, 32)
        label_embeddings = embedding_layer(labels).detach().numpy()
        adata_omics2.obsm['label_embeddings_AE'] = label_embeddings
        emb_latent_omics2 = adata_omics2.obsm['label_embeddings_AE']
    else:
        emb_latent_omics2 = adata_omics2.obsm['emb_latent_omics2']

    # Spatial alignment (consider batch effects in horizontal integration)
    # FIX: Use positional alignment instead of tuple-index reindex to avoid NaN
    # When cells come from the same fusion data, obs_names should already be aligned,
    # so reindex by tuple coords is fragile (float precision). Use direct alignment.
    N_cells = adata_omics1.n_obs
    if emb_latent_omics1.shape[0] == emb_latent_omics2.shape[0] == N_cells:
        # Same cells in same order -- no reindex needed
        print(f"[PATCH] Both embeddings have {N_cells} cells, skipping spatial reindex (avoids NaN)")
        aligned_emb_latent_omics1 = emb_latent_omics1.copy()
        aligned_emb_latent_omics2 = emb_latent_omics2.copy()
    else:
        # Fallback: original reindex but with NaN safety
        df_omics1 = pd.DataFrame(emb_latent_omics1, index=[tuple(coord) for coord in slices_omics1_spatial])
        df_omics2 = pd.DataFrame(emb_latent_omics2, index=[tuple(coord) for coord in slices_omics2_spatial])
        df_omics2_aligned = df_omics2.reindex(df_omics1.index)
        aligned_emb_latent_omics1 = df_omics1.to_numpy()
        aligned_emb_latent_omics2 = df_omics2_aligned.to_numpy()

    # FIX: Replace any NaN with 0 to prevent NaN propagation in diffusion training
    nan_count = np.isnan(aligned_emb_latent_omics2).sum()
    if nan_count > 0:
        print(f"[PATCH] Replacing {nan_count} NaN values in aligned_emb_latent_omics2 with 0")
        aligned_emb_latent_omics2 = np.nan_to_num(aligned_emb_latent_omics2, nan=0.0)
    nan_count1 = np.isnan(aligned_emb_latent_omics1).sum()
    if nan_count1 > 0:
        print(f"[PATCH] Replacing {nan_count1} NaN values in aligned_emb_latent_omics1 with 0")
        aligned_emb_latent_omics1 = np.nan_to_num(aligned_emb_latent_omics1, nan=0.0)

    print(f"Aligned embeddings shapes: {aligned_emb_latent_omics1.shape}, {aligned_emb_latent_omics2.shape}")

    # Choose denoising target for horizontal integration
    if modality == 'ADT':
        dataset = EfficientConditionalDiffusionDataset(aligned_emb_latent_omics1, aligned_emb_latent_omics2)
        denoise_target = 'omics1'
    else:
        dataset = EfficientConditionalDiffusionDataset(aligned_emb_latent_omics2, aligned_emb_latent_omics1)
        denoise_target = 'omics2'

    # FIX: Reduce epochs and folds for large datasets to avoid timeout
    is_large = N_cells > _LARGE_DATASET_THRESHOLD
    if is_large:
        diff_k = 2  # KFold requires n_splits >= 2
        diff_epochs = 300
        diff_patience = 30
        print(f"[PATCH] Large dataset ({N_cells} cells): k={diff_k}, epochs={diff_epochs}, patience={diff_patience}")
    else:
        diff_k = 3
        diff_epochs = 1000
        diff_patience = 40

    # Run diffusion denoising (adapted for batch effect removal)
    seed_everything(2024)
    com_mtx = run_diff(
        dataset,
        k=diff_k,
        batch_size=min(512, max(32, adata_omics1.n_obs//4)),
        hidden_size=256,
        learning_rate=1e-3,
        num_epoch=diff_epochs,
        diffusion_step=800,
        depth=6,
        head=16,
        pca_dim=50,
        device=device.type,
        classes=6,
        patience=diff_patience,
        bias=0.7 if modality == 'ADT' else 1.2  # Adjusted for horizontal integration
    )

    # Store denoised embeddings
    if denoise_target == 'omics1':
        adata_omics1.obsm['denoise_emb'] = com_mtx
    else:
        adata_omics2.obsm['denoise_emb'] = com_mtx

    # === Integration phase (horizontal integration specific) ===
    print("Integration phase for horizontal integration...")
    
    adata1 = adata_omics1.copy()
    adata2 = adata_omics2.copy()
    
    # Spatial alignment for horizontal integration
    spatial1 = pd.DataFrame(adata1.obsm['spatial'], columns=['x', 'y'])
    spatial2 = pd.DataFrame(adata2.obsm['spatial'], columns=['x', 'y'])
    spatial1['index1'] = spatial1.index
    spatial2['index2'] = spatial2.index
    
    print(f"Before spatial alignment - adata1: {adata1.n_obs} cells, adata2: {adata2.n_obs} cells")
    print(f"Spatial1 shape: {spatial1.shape}, Spatial2 shape: {spatial2.shape}")
    
    merged = pd.merge(spatial1, spatial2, on=['x', 'y'], how='inner')
    print(f"After spatial merge - merged shape: {merged.shape}")
    
    if len(merged) == 0:
        print("Warning: No common spatial coordinates found! Using original data without spatial alignment.")
        # Keep original data without spatial filtering
        pass
    else:
        sorted_index1 = merged['index1'].values
        sorted_index2 = merged['index2'].values
        adata1 = adata1[sorted_index1]
        adata2 = adata2[sorted_index2]
        print(f"After spatial alignment - adata1: {adata1.n_obs} cells, adata2: {adata2.n_obs} cells")

    # Set features for integration (prioritize denoised features)
    if denoise_target == 'omics1':
        adata1.obsm['feat'] = adata1.obsm['denoise_emb']
        adata2.obsm['feat'] = adata2.obsm['emb_latent_omics2']
    else:
        adata1.obsm['feat'] = adata1.obsm['emb_ZINB']
        adata2.obsm['feat'] = adata2.obsm['denoise_emb']

    # Reconstruct neighbor graphs for final integration
    adata1, adata2 = construct_neighbor_graph(adata1, adata2)
    adj = adjacent_matrix_preprocessing(adata1, adata2)
    adj_spatial_omics1 = adj['adj_spatial_omics1'].to(device)
    adj_spatial_omics2 = adj['adj_spatial_omics2'].to(device)
    adj_feature_omics1 = adj['adj_feature_omics1'].to(device)
    adj_feature_omics2 = adj['adj_feature_omics2'].to(device)

    features_omics1 = torch.FloatTensor(adata1.obsm['feat'].copy()).to(device)
    features_omics2 = torch.FloatTensor(adata2.obsm['feat'].copy()).to(device)

    # Final integration training (batch effect removal)
    # FIX: Use patched train_and_infer with SampledNTXentLoss for large datasets
    seed_everything(2025)
    result = _patched_train_and_infer(
        features_omics1=features_omics1,
        features_omics2=features_omics2,
        adj_spatial_omics1=adj_spatial_omics1,
        adj_feature_omics1=adj_feature_omics1,
        adj_spatial_omics2=adj_spatial_omics2,
        adj_feature_omics2=adj_feature_omics2,
        device=device,
        epochs=250  # Increased epochs for horizontal integration
    )
    
    # === End training time recording ===
    end_time = time.time()
    train_time = end_time - start_time
    print('Total horizontal integration training time:', train_time)

    # === Build Result AnnData ===
    adata = adata1.copy()
    adata.obsm['CANDIES'] = result['emb_latent_combined'].detach().cpu().numpy().copy()
    adata.uns['train_time'] = train_time
    adata.uns['integration_type'] = 'horizontal'
    
    # === Parse Dataset Info ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Detected dataset: {dataset_name}, subset: {subset_name}")

    # === Plot Save Path ===
    plot_base_dir = "Results/plot"
    method_name = args.method if args.method else "CANDIES"
    plot_dir = os.path.join(plot_base_dir, "horizontal_integration", method_name, dataset_name, subset_name)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plot images will be saved to: {plot_dir}")

    # === Clustering and Visualization ===
    tools = ['mclust', 'louvain', 'leiden', 'kmeans']
    
    # === Generate UMAP for visualization (only once) ===
    sc.pp.neighbors(adata, use_rep='CANDIES', n_neighbors=30)
    sc.tl.umap(adata)
    
    import sys
    for tool in tools:
        print(f"[DEBUG] === Starting clustering tool: {tool} ===", flush=True)
        sys.stdout.flush()
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='CANDIES',
            method=tool,
            key=tool,
            use_pca=False
        )
        print(f"[DEBUG] === Finished clustering tool: {tool}, starting visualization ===", flush=True)
        sys.stdout.flush()

        # Flip spatial coordinates for visualization if available
        if 'spatial' in adata.obsm.keys():
            adata.obsm['spatial'][:, 1] = -1 * adata.obsm['spatial'][:, 1]

        fig, ax_list = plt.subplots(1, 3, figsize=(10, 3))

        # Plot UMAP, spatial, and batch
        sc.pl.umap(adata, color=tool, ax=ax_list[0], title=f'{method_name}-{tool}', s=20, show=False)
        if 'spatial' in adata.obsm.keys():
            sc.pl.embedding(adata, basis='spatial', color=tool, ax=ax_list[1], title=f'{method_name}-{tool}', s=20, show=False)
        else:
            sc.pl.umap(adata, color=tool, ax=ax_list[1], title=f'{method_name}-{tool} (no spatial)', s=20, show=False)
        
        # Plot batch distribution for horizontal integration evaluation
        if 'batch' in adata.obs.columns:
            sc.pl.umap(adata, color='batch', ax=ax_list[2], title=f'{method_name} batch', s=20, show=False)
        else:
            sc.pl.umap(adata, color=tool, ax=ax_list[2], title=f'{method_name}-{tool} (no batch)', s=20, show=False)

        plt.tight_layout(w_pad=0.3)
        plt.savefig(
            os.path.join(plot_dir, f'clustering_{tool}_umap_spatial_batch.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    # === Save AnnData ===
    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir, exist_ok=True)
    adata.write(args.save_path)
    print(adata)
    print('Saving horizontal integration results to...', args.save_path)

    # === Save Timing JSON ===
    total_time = time.time() - total_start_time
    dataset_name, _ = parse_dataset_info(args)
    modality_str = "RNA_ADT" if args.ADT_path else "RNA_ATAC"
    timing_info = {
        "method": "CANDIES",
        "dataset": dataset_name,
        "integration_type": "horizontal",
        "modality": modality_str,
        "n_cells": adata.n_obs,
        "embedding_shape": list(adata.obsm["CANDIES"].shape),
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


if __name__ == "__main__":
    # Set environment variables for R and threading
    os.environ['R_HOME'] = '/home/users/nus/e1724738/miniconda3/envs/_Proj1_1/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    print("Starting CANDIES horizontal integration...")
    parser = argparse.ArgumentParser(description='Run CANDIES horizontal integration')
    parser.add_argument('--data_type', type=str, default='fusion', help='Data type for horizontal integration')
    parser.add_argument('--RNA_path', type=str, required=True, help='Path to RNA fusion adata')
    parser.add_argument('--ADT_path', type=str, default='', help='Path to ADT fusion adata')
    parser.add_argument('--ATAC_path', type=str, default='', help='Path to ATAC fusion adata')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save integrated adata')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, e.g. cuda:0 or cpu')

    parser.add_argument('--method', type=str, default='CANDIES', help='Method name for plotting')
    parser.add_argument('--dataset', type=str, default='', help='Dataset name for horizontal integration')

    parser.add_argument('--cluster_nums', type=int, help='Number of clusters')

    args = parser.parse_args()
    main(args)
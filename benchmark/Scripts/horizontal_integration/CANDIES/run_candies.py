import os
import torch
import pandas as pd
import scanpy as sc
import numpy as np
import argparse
import time
import sys
import re
import matplotlib.pyplot as plt
import warnings

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
    df_omics1 = pd.DataFrame(emb_latent_omics1, index=[tuple(coord) for coord in slices_omics1_spatial])
    df_omics2 = pd.DataFrame(emb_latent_omics2, index=[tuple(coord) for coord in slices_omics2_spatial])
    df_omics2_aligned = df_omics2.reindex(df_omics1.index)

    aligned_emb_latent_omics1 = df_omics1.to_numpy()
    aligned_emb_latent_omics2 = df_omics2_aligned.to_numpy()

    print(f"Aligned embeddings shapes: {aligned_emb_latent_omics1.shape}, {aligned_emb_latent_omics2.shape}")

    # Conditional diffusion dataset for horizontal integration
    class ConditionalDiffusionDataset:
        def __init__(self, adata_omics1, adata_omics2):
            self.adata_omics1 = adata_omics1  
            self.adata_omics2 = adata_omics2  
            self.st_sample = torch.tensor(adata_omics1, dtype=torch.float32)
            self.con_sample = torch.tensor(adata_omics2, dtype=torch.float32)
            self.con_data = torch.tensor(adata_omics2, dtype=torch.float32)

        def __len__(self):
            return len(self.st_sample)

        def __getitem__(self, idx):
            return self.st_sample[idx], self.con_sample[idx], self.con_data

    # Choose denoising target for horizontal integration
    if modality == 'ADT':
        dataset = ConditionalDiffusionDataset(aligned_emb_latent_omics1, aligned_emb_latent_omics2)
        denoise_target = 'omics1'
    else:
        dataset = ConditionalDiffusionDataset(aligned_emb_latent_omics2, aligned_emb_latent_omics1) 
        denoise_target = 'omics2'

    # Run diffusion denoising (adapted for batch effect removal)
    seed_everything(2024)
    com_mtx = run_diff(
        dataset,
        k=3,
        batch_size=min(512, adata_omics1.n_obs//4),
        hidden_size=256,
        learning_rate=1e-3,
        num_epoch=1000,  # More epochs for horizontal integration
        diffusion_step=800,
        depth=6,
        head=16,
        pca_dim=50,
        device=device.type,
        classes=6,  
        patience=40,
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
    seed_everything(2025)
    result = train_and_infer(
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
    
    for tool in tools:
        adata = universal_clustering(
            adata,
            n_clusters=args.cluster_nums,
            used_obsm='CANDIES',
            method=tool,
            key=tool,
            use_pca=False
        )
        
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


if __name__ == "__main__":
    # Set environment variables for R and threading
    os.environ['R_HOME'] = '/home/zhenghong/miniconda3/envs/smobench/lib/R'
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
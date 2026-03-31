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
    """Extract dataset_name and subset_name from RNA_path or save_path"""
    if hasattr(args, 'dataset') and args.dataset:
        parts = args.dataset.strip('/').split('/')
        if len(parts) == 2:
            return parts[0], parts[1]
        elif len(parts) == 1:
            return parts[0], "Unknown"
    
    # Auto parse RNA_path
    match = re.search(r'Dataset/([^/]+)/([^/]+)/([^/]+)/adata_RNA\.h5ad', args.RNA_path)
    if match:
        return match.group(2), match.group(3)
    return "Unknown", "Unknown"


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
    
    # === Load Data ===
    adata_omics1 = sc.read_h5ad(args.RNA_path)  # RNA data
    adata_omics1.var_names_make_unique()

    if args.ADT_path:
        adata_omics2 = sc.read_h5ad(args.ADT_path)  # ADT data
        modality = 'ADT'
        modality_name = 'Proteome'
        adata_omics2.var_names_make_unique()
    elif args.ATAC_path:
        adata_omics2 = sc.read_h5ad(args.ATAC_path)  # ATAC data
        modality = 'ATAC'
        modality_name = 'Epigenome'
        adata_omics2.var_names_make_unique()
    else:
        raise ValueError("Either ADT_path or ATAC_path must be provided.")

    print(f"Processing {args.data_type}: RNA + {modality_name} integration...")

    # === Ensure both datasets have same cells ===
    common_obs = adata_omics1.obs_names.intersection(adata_omics2.obs_names)
    print(f"Common cells: {len(common_obs)}, RNA cells: {adata_omics1.n_obs}, {modality_name} cells: {adata_omics2.n_obs}")
    adata_omics1 = adata_omics1[common_obs].copy()
    adata_omics2 = adata_omics2[common_obs].copy()
    print(f"After intersection - RNA: {adata_omics1.n_obs}, {modality_name}: {adata_omics2.n_obs}")

    # === Data preprocessing ===
    print("Data preprocessing...")
    
    # RNA preprocessing
    sc.pp.filter_genes(adata_omics1, min_cells=10)
    sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1)

    if modality == 'ADT':
        # Protein preprocessing
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars-1)
        
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars-1)
        
    else:  # ATAC preprocessing
        if 'X_lsi' not in adata_omics2.obsm.keys():
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
            lsi(adata_omics2, use_highly_variable=False, n_components=51)
        
        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.obsm['X_lsi'].shape[1])

    # === Start training time recording ===
    start_time = time.time()
    
    # === Encoding phase ===
    print("Encoding phase...")
    
    # Construct neighbor graphs
    adata_omics1, adata_omics2 = construct_neighbor_graph(adata_omics1, adata_omics2)
    adj = adjacent_matrix_preprocessing(adata_omics1, adata_omics2)
    adj_spatial_omics1 = adj['adj_spatial_omics1'].to(device)
    adj_spatial_omics2 = adj['adj_spatial_omics2'].to(device)

    # RNA encoding with ZINB
    ae_model = encoder_ZINB(
        adata=adata_omics1,
        device=device,
        epochs=300, 
        dim_output=64 if modality == 'ADT' else 128
    )
    adata_omics1.obsm['emb_ZINB'], adj_mat = ae_model.train()

    # Second modality encoding
    seed_everything(args.seed)
    if modality == 'ADT':
        train_model(adata_omics1, adata_omics2, adj_spatial_omics1, adj_spatial_omics2, epochs=400)
    else:  # ATAC
        train_atac(adata_omics2, adj_spatial_omics2, epochs=600)

    # === Pre-integration clustering (respecting tutorial approach) ===
    print("Pre-integration clustering...")
    if modality == 'ADT':
        cluster_num = min(10, args.cluster_nums)
        adata_omics1 = run_leiden_candies(adata_omics1, n_cluster=cluster_num, use_rep="emb_ZINB", key_added="AE")
        adata_omics2 = run_leiden_candies(adata_omics2, n_cluster=cluster_num, use_rep="emb_latent_omics2", key_added="AE")
    else:  # RNA+ATAC
        cluster_num = min(14, args.cluster_nums)
        adata_omics1 = run_leiden_candies(adata_omics1, n_cluster=cluster_num, use_rep="emb_ZINB", key_added="AE")
        adata_omics2 = clustering_candies(adata_omics2, key='emb_latent_omics2', add_key='AE', n_clusters=cluster_num, method='mclust')

    # === Modality selection ===
    print("Modality selection...")
    try:
        modality_selection(adata_omics1.obs['AE'], adata_omics2.obs['AE'], 
                         modality1_embs=adata_omics1.obsm['emb_ZINB'], 
                         modality2_embs=adata_omics2.obsm['emb_latent_omics2'], 
                         spatial_coor=adata_omics1.obsm['spatial'])
    except:
        print("Modality selection completed (using default recommendation)")

    # === Denoise phase ===
    print("Denoise phase...")
    
    # Align embeddings according to spatial coordinates  
    slices_omics1_spatial = adata_omics1.obsm['spatial']
    slices_omics2_spatial = adata_omics2.obsm['spatial']
    emb_latent_omics1 = adata_omics1.obsm['emb_ZINB']
    
    if modality == 'ADT':
        # For ADT: create label embeddings from clusters
        ad2_ae = adata_omics2.obs['AE']
        labels = torch.tensor(ad2_ae.values.astype(int), dtype=torch.long)
        num_classes = labels.max().item() + 1 
        embedding_layer = torch.nn.Embedding(num_classes, 32)
        label_embeddings = embedding_layer(labels).detach().numpy()
        adata_omics2.obsm['label_embeddings_AE'] = label_embeddings
        emb_latent_omics2 = adata_omics2.obsm['label_embeddings_AE']
    else:
        emb_latent_omics2 = adata_omics2.obsm['emb_latent_omics2']

    df_omics1 = pd.DataFrame(emb_latent_omics1, index=[tuple(coord) for coord in slices_omics1_spatial])
    df_omics2 = pd.DataFrame(emb_latent_omics2, index=[tuple(coord) for coord in slices_omics2_spatial])
    df_omics2_aligned = df_omics2.reindex(df_omics1.index)

    aligned_emb_latent_omics1 = df_omics1.to_numpy()
    aligned_emb_latent_omics2 = df_omics2_aligned.to_numpy()

    print(f"Aligned embeddings shapes: {aligned_emb_latent_omics1.shape}, {aligned_emb_latent_omics2.shape}")

    # Conditional diffusion dataset
    class ConditionalDiffusionDataset:
        def __init__(self, adata_omics1, adata_omics2):
            self.adata_omics1 = adata_omics1  # Add this for compatibility with run_diff
            self.adata_omics2 = adata_omics2  # Add this for compatibility with run_diff
            self.st_sample = torch.tensor(adata_omics1, dtype=torch.float32)
            self.con_sample = torch.tensor(adata_omics2, dtype=torch.float32)
            self.con_data = torch.tensor(adata_omics2, dtype=torch.float32)

        def __len__(self):
            return len(self.st_sample)

        def __getitem__(self, idx):
            return self.st_sample[idx], self.con_sample[idx], self.con_data

    # Choose which modality to denoise based on tutorial
    if modality == 'ADT':
        dataset = ConditionalDiffusionDataset(aligned_emb_latent_omics1, aligned_emb_latent_omics2)
        denoise_target = 'omics1'
    else:
        dataset = ConditionalDiffusionDataset(aligned_emb_latent_omics2, aligned_emb_latent_omics1) 
        denoise_target = 'omics2'

    # Run diffusion denoising
    seed_everything(2024)
    com_mtx = run_diff(
        dataset,
        k=3,
        batch_size=min(512, adata_omics1.n_obs//4),
        hidden_size=256,
        learning_rate=1e-3,
        num_epoch=1000,
        diffusion_step=800,
        depth=6,
        head=16,
        pca_dim=50,
        device=device.type,
        classes=6,  
        patience=40,
        bias=0.5 if modality == 'ADT' else 1
    )

    # Store denoised embeddings
    if denoise_target == 'omics1':
        adata_omics1.obsm['denoise_emb'] = com_mtx
    else:
        adata_omics2.obsm['denoise_emb'] = com_mtx

    # === Integration phase ===
    print("Integration phase...")
    
    adata1 = adata_omics1.copy()
    adata2 = adata_omics2.copy()
    
    # Spatial alignment
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

    # Set features for integration
    if denoise_target == 'omics1':
        adata1.obsm['feat'] = adata1.obsm['denoise_emb']
        adata2.obsm['feat'] = adata2.obsm['emb_latent_omics2']
    else:
        adata1.obsm['feat'] = adata1.obsm['emb_ZINB']
        adata2.obsm['feat'] = adata2.obsm['denoise_emb']

    # Reconstruct neighbor graphs for integration
    adata1, adata2 = construct_neighbor_graph(adata1, adata2)
    adj = adjacent_matrix_preprocessing(adata1, adata2)
    adj_spatial_omics1 = adj['adj_spatial_omics1'].to(device)
    adj_spatial_omics2 = adj['adj_spatial_omics2'].to(device)
    adj_feature_omics1 = adj['adj_feature_omics1'].to(device)
    adj_feature_omics2 = adj['adj_feature_omics2'].to(device)

    features_omics1 = torch.FloatTensor(adata1.obsm['feat'].copy()).to(device)
    features_omics2 = torch.FloatTensor(adata2.obsm['feat'].copy()).to(device)

    # Final integration training
    seed_everything(2025)
    result = train_and_infer(
        features_omics1=features_omics1,
        features_omics2=features_omics2,
        adj_spatial_omics1=adj_spatial_omics1,
        adj_feature_omics1=adj_feature_omics1,
        adj_spatial_omics2=adj_spatial_omics2,
        adj_feature_omics2=adj_feature_omics2,
        device=device,
        epochs=200
    )
    
    # === End training time recording ===
    end_time = time.time()
    train_time = end_time - start_time
    print('Total training time:', train_time)

    # === Build Result AnnData ===
    adata = adata1.copy()
    adata.obsm['CANDIES'] = result['emb_latent_combined'].detach().cpu().numpy().copy()
    adata.uns['train_time'] = train_time
    
    # === Parse Dataset Info ===
    dataset_name, subset_name = parse_dataset_info(args)
    print(f"Detected dataset: {dataset_name}, subset: {subset_name}")

    # === Plot Save Path ===
    plot_base_dir = "Results/plot"
    method_name = args.method if args.method else "CANDIES"
    plot_dir = os.path.join(plot_base_dir, method_name, dataset_name, subset_name)
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

        fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))

        # Plot UMAP and spatial
        sc.pl.umap(adata, color=tool, ax=ax_list[0], title=f'{method_name}-{tool}', s=20, show=False)
        if 'spatial' in adata.obsm.keys():
            sc.pl.embedding(adata, basis='spatial', color=tool, ax=ax_list[1], title=f'{method_name}-{tool}', s=20, show=False)
        else:
            # If no spatial coordinates, plot UMAP again
            sc.pl.umap(adata, color=tool, ax=ax_list[1], title=f'{method_name}-{tool} (no spatial)', s=20, show=False)

        plt.tight_layout(w_pad=0.3)
        plt.savefig(
            os.path.join(plot_dir, f'clustering_{tool}_umap_spatial.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    # === Save AnnData ===
    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir, exist_ok=True)
    adata.write(args.save_path)
    print(adata)
    print('Saving results to...', args.save_path)


if __name__ == "__main__":
    # Set environment variables for R and threading
    os.environ['R_HOME'] = '/home/zhenghong/miniconda3/envs/smobench/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    print("Starting CANDIES integration...")
    parser = argparse.ArgumentParser(description='Run CANDIES integration')
    parser.add_argument('--data_type', type=str, default='10x', help='Data type, e.g. 10x, SPOTS, MISAR, simulation')
    parser.add_argument('--RNA_path', type=str, required=True, help='Path to RNA adata')
    parser.add_argument('--ADT_path', type=str, default='', help='Path to ADT adata')
    parser.add_argument('--ATAC_path', type=str, default='', help='Path to ATAC adata')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save integrated adata')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, e.g. cuda:0 or cpu')

    parser.add_argument('--method', type=str, default='CANDIES', help='Method name for plotting')
    parser.add_argument('--dataset', type=str, default='', help='Dataset name, e.g. Human_Lymph_Nodes/A1. If not provided, auto-extracted from paths.')

    parser.add_argument('--cluster_nums', type=int, help='Number of clusters')

    args = parser.parse_args()
    main(args)
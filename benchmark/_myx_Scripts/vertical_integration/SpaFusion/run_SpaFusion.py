#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_SpaFusion_vertical.py
"""

import os
import sys
import time
import torch
import argparse
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================
# 路径设置
# ==============================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

spafusion_path = os.path.join(project_root, "Methods/SpaFusion")
sys.path.append(spafusion_path)

from main import pre_train, train, setup_seed, load_data, norm_adj, adjacent_matrix_preprocessing
from high_order_matrix import process_adjacency_matrix
from Utils.SMOBench_clustering import universal_clustering


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("SpaFusion vertical integration running on:", device)

    setup_seed(args.seed)

    adata_rna = sc.read_h5ad(args.RNA_path)
    adata_rna.var_names_make_unique()

    if args.ADT_path:
        adata_other = sc.read_h5ad(args.ADT_path)
        modality = "ADT"
        view2 = "Protein"   # ✅ 改为 SpaFusion 内部识别的标签
    elif args.ATAC_path:
        adata_other = sc.read_h5ad(args.ATAC_path)
        modality = "ATAC"
        view2 = "Chromatin"  # ✅ 改为 SpaFusion 内部识别的标签
    else:
        raise ValueError("Must provide either ADT_path or ATAC_path.")

    # 然后这样调用
    adata_rna, adata_other = load_data(
        adata_omics1=adata_rna,
        view1="RNA",
        adata_omics2=adata_other,
        view2=view2,              # ✅ 改成 view2 而不是 modality
        n_neighbors=args.spatial_k,
        k=args.adj_k
    )


    data1 = adata_rna.obsm['feat']
    data2 = adata_other.obsm['feat']

    pre_adj_dir = Path("./pre_adj")
    pre_adj_dir.mkdir(parents=True, exist_ok=True)

    adj = adjacent_matrix_preprocessing(adata_rna, adata_other, Path("./pre_adj"))

    feature_adj1 = norm_adj(adj['adj_feature_omics1'])
    feature_adj2 = norm_adj(adj['adj_feature_omics2'])
    spatial_adj1 = norm_adj(adj['adj_spatial_omics1'])
    spatial_adj2 = norm_adj(adj['adj_spatial_omics2'])

    Mt1_path = pre_adj_dir / f"Mt1_{args.dataset}.npy"
    Mt2_path = pre_adj_dir / f"Mt2_{args.dataset}.npy"

    Mt1 = norm_adj(process_adjacency_matrix(feature_adj1, Mt1_path))
    Mt2 = norm_adj(process_adjacency_matrix(feature_adj2, Mt2_path))

    tensors = lambda x: torch.tensor(x, dtype=torch.float32).to(device)
    data1, data2 = tensors(data1), tensors(data2)
    feature_adj1, feature_adj2 = tensors(feature_adj1), tensors(feature_adj2)
    spatial_adj1, spatial_adj2 = tensors(spatial_adj1), tensors(spatial_adj2)
    Mt1, Mt2 = tensors(Mt1), tensors(Mt2)

    print("Starting SpaFusion pretraining...")
    pre_start = time.time()
    # print(">>> RNA:", adata_rna.shape, "ADT:", adata_other.shape)
    # print(">>> RNA obs head:", adata_rna.obs_names[:5])
    # print(">>> ADT obs head:", adata_other.obs_names[:5])


    emb_comb, emb1, emb2 = pre_train(
        x1=data1, x2=data2,
        spatial_adj1=spatial_adj1, feature_adj1=feature_adj1,
        spatial_adj2=spatial_adj2, feature_adj2=feature_adj2,
        Mt1=Mt1, Mt2=Mt2,
        y=None, n_clusters=args.cluster_nums,
        num_epoch=args.pretrain_epoch,
        device=device, lr=args.lr,
        dataset_name=args.dataset, weight_list=args.weight_list
    )
    pre_time = time.time() - pre_start

    print("Starting SpaFusion training...")
    train_start = time.time()
    emb_comb, emb1, emb2 = train(
        x1=data1, x2=data2,
        spatial_adj1=spatial_adj1, feature_adj1=feature_adj1,
        spatial_adj2=spatial_adj2, feature_adj2=feature_adj2,
        y=None, n_clusters=args.cluster_nums,
        Mt1=Mt1, Mt2=Mt2,
        num_epoch=args.train_epoch,
        lambda1=args.lambda1, lambda2=args.lambda2,
        device=device, seed=args.seed, lr=args.lr,num=0,
        dataset_name=args.dataset, weight_list=args.weight_list,
        spatial_K=args.spatial_k, adj_K=args.adj_k
    )
    train_time = time.time() - train_start

    adata = adata_rna.copy()
    adata.obsm['SpaFusion'] = emb_comb.detach().cpu().numpy()
    adata.obsm['emb_latent_omics1'] = emb1.detach().cpu().numpy()
    adata.obsm['emb_latent_omics2'] = emb2.detach().cpu().numpy()

    adata.uns.update({
        'train_time': train_time,
        'pretrain_time': pre_time,
        'integration_type': 'vertical',
        'modality': modality
    })

    sc.pp.neighbors(adata, use_rep='SpaFusion', n_neighbors=30)
    sc.tl.umap(adata)

    for tool in ['mclust', 'louvain', 'leiden', 'kmeans']:
        adata = universal_clustering(adata, n_clusters=args.cluster_nums, used_obsm='SpaFusion', method=tool, key=tool)

    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir, exist_ok=True)
    adata.write(args.save_path)

    print(adata)
    print(f"SpaFusion vertical integration completed: {args.save_path}")


if __name__ == '__main__':
    os.environ['R_HOME'] = '/home/zhenghong/miniconda3/envs/smobench/lib/R'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description='Run SpaFusion Vertical Integration (RNA+ADT/ATAC)')
    parser.add_argument('--RNA_path', type=str, required=True)
    parser.add_argument('--ADT_path', type=str, default='')
    parser.add_argument('--ATAC_path', type=str, default='')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--cluster_nums', type=int, default=10)

    parser.add_argument('--spatial_k', type=int, default=9)
    parser.add_argument('--adj_k', type=int, default=20)

    parser.add_argument('--pretrain_epoch', type=int, default=100)
    parser.add_argument('--train_epoch', type=int, default=100)

    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--lambda2', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_list', type=list, default=[1,1,1,1,1,1])

    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    main(args)

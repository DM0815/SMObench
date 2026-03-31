#!/usr/bin/env python3
"""
SMOBench 3M (Three-Modality) Integration Evaluation
Computes SC + BioC + CM-GTC for SpaBalance_3M, PRAGA_3M, SpatialGlue_3M
on the 3M simulation dataset (RNA + ADT + ATAC).

Data:
  GT:        Dataset/withGT/3M_Simulation/adata_{RNA,ADT,ATAC}.h5ad
  Results:   _myx_Results/adata/vertical_integration/{Method}_3M/

Usage:
    python eval_3m.py --root /path/to/SMOBench-CLEAN
    python eval_3m.py --root /path/to/SMOBench-CLEAN --test
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

METHODS_3M = ['PRAGA_3M', 'SpaBalance_3M', 'SpatialGlue_3M']

METHODS_3Mv2 = [
    'PRAGA_3Mv2', 'SpaBalance_3Mv2', 'SpatialGlue_3Mv2',
    'PRESENT_3Mv2', 'SpaMV_3Mv2', 'MISO_3Mv2', 'SMOPCA_3Mv2',
]

# Map method name → embedding key alternatives
METHOD_EMBEDDING_KEYS = {
    'PRAGA_3M':        ['PRAGA_3M', 'PRAGA', 'X_integrated', 'X_emb'],
    'SpaBalance_3M':   ['SpaBalance_3M', 'SpaBalance', 'X_integrated', 'X_emb'],
    'SpatialGlue_3M':  ['SpatialGlue_3M', 'SpatialGlue', 'X_integrated', 'X_emb'],
    # v2 methods
    'PRAGA_3Mv2':      ['PRAGA', 'PRAGA_3Mv2', 'X_integrated', 'X_emb'],
    'SpaBalance_3Mv2': ['SpaBalance', 'SpaBalance_3Mv2', 'X_integrated', 'X_emb'],
    'PRESENT_3Mv2':    ['PRESENT', 'PRESENT_3Mv2', 'embeddings', 'X_integrated', 'X_emb'],
    'SpaMV_3Mv2':      ['SpaMV', 'SpaMV_3Mv2', 'X_integrated', 'X_emb'],
    'MISO_3Mv2':       ['MISO', 'MISO_3Mv2', 'X_integrated', 'X_emb'],
    'SMOPCA_3Mv2':     ['SMOPCA', 'SMOPCA_3Mv2', 'X_integrated', 'X_emb'],
    'SpatialGlue_3Mv2': ['SpatialGlue', 'SpatialGlue_3Mv2', 'X_integrated', 'X_emb'],
}

CLUSTERING_METHODS = ['leiden', 'louvain', 'kmeans', 'mclust']

GT_DIR = 'Dataset/withGT/3M_Simulation'
GT_DIR_V2 = 'Dataset/withGT/3M_Simulation_v2'


def parse_args():
    parser = argparse.ArgumentParser(description='SMOBench 3M Evaluation')
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--methods', nargs='+', default=None)
    parser.add_argument('--clustering', nargs='+', default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--version', type=str, default='v1', choices=['v1', 'v2'],
                        help='v1=original 3M_Simulation, v2=3M_Simulation_v2 with new data')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_dense(X):
    if sparse.issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X)


def setup_eval_imports(root_dir):
    """Import core eval functions from Eval/."""
    eval_dir = os.path.join(root_dir, 'Eval')
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)

    from src.demo import eval_vertical_integration, save_evaluation_results
    try:
        from src.clustering import knn_adj_matrix
    except ImportError:
        from src.clustering_simple import knn_adj_matrix

    return eval_vertical_integration, save_evaluation_results, knn_adj_matrix


def setup_cmgtc_import(root_dir):
    """Import CMGTC from storage."""
    candidates = [
        os.path.join(os.path.dirname(root_dir), 'storage', '_2_metric'),
        '/home/users/nus/e1724738/_main/_private/NUS/_Proj1/storage/_2_metric',
    ]
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, 'cm_gtc.py')):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            from cm_gtc import CMGTC
            return CMGTC
    print("WARNING: cm_gtc.py not found, CM-GTC will be skipped")
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_3m_gt(root_dir, version='v1'):
    """Load GT labels and modality data for 3M simulation."""
    gt_dir = GT_DIR_V2 if version == 'v2' else GT_DIR
    gt_base = os.path.join(root_dir, gt_dir)
    data = {}

    for mod in ['RNA', 'ADT', 'ATAC']:
        path = os.path.join(gt_base, f'adata_{mod}.h5ad')
        if os.path.isfile(path):
            adata = sc.read_h5ad(path)
            data[f'adata_{mod.lower()}'] = adata
            data[mod.lower()] = _to_dense(adata.X)
            print(f"  Loaded {mod}: {adata.shape}")

            # Extract GT labels (try common column names)
            if 'gt_labels' not in data:
                for col in ['Spatial_Label', 'Ground Truth', 'cell_type', 'cluster']:
                    if col in adata.obs.columns:
                        data['gt_labels'] = adata.obs[col].values
                        data['gt_col'] = col
                        print(f"  GT labels from {col}: {len(np.unique(data['gt_labels']))} types")
                        break
        else:
            print(f"  WARNING: {path} not found")

    return data


def load_3m_result(root_dir, method):
    """Load 3M method result h5ad."""
    result_dir = os.path.join(root_dir, '_myx_Results', 'adata',
                               'vertical_integration', method)
    if not os.path.isdir(result_dir):
        return None

    h5ad_files = sorted(Path(result_dir).rglob('*.h5ad'))
    if not h5ad_files:
        return None

    return sc.read_h5ad(str(h5ad_files[0]))


def get_embedding(adata, method):
    """Extract embedding from adata.obsm."""
    keys = METHOD_EMBEDDING_KEYS.get(method, [method, 'X_integrated', 'X_emb'])
    for key in keys:
        if key in adata.obsm:
            return np.asarray(adata.obsm[key]), key
    return None, None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_3m(root_dir, methods, clustering_methods, test_mode=False, version='v1'):
    """Run SC + BioC + CM-GTC evaluation for 3M methods."""

    eval_func, save_func, knn_adj = setup_eval_imports(root_dir)
    CMGTC_cls = setup_cmgtc_import(root_dir)

    out_suffix = '_v2' if version == 'v2' else ''
    output_dir = os.path.join(root_dir, '_myx_Results', 'evaluation', f'3m{out_suffix}')
    os.makedirs(output_dir, exist_ok=True)

    # Load GT data (shared across all methods)
    print(f"\n--- Loading 3M GT data (version={version}) ---")
    gt_data = load_3m_gt(root_dir, version=version)
    if not gt_data:
        print("ERROR: No GT data found for 3M simulation")
        return []

    # Prepare GT labels
    y_GT = None
    if 'gt_labels' in gt_data:
        y_GT_raw = gt_data['gt_labels']
        # Convert to int codes if categorical/string
        if hasattr(y_GT_raw, 'astype'):
            try:
                y_GT = y_GT_raw.astype(int)
            except (ValueError, TypeError):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_GT = le.fit_transform(y_GT_raw)

    # Prepare CM-GTC evaluator
    cmgtc_evaluator = None
    if CMGTC_cls is not None:
        cmgtc_evaluator = CMGTC_cls(
            similarity_metric='cosine',
            correlation_metric='spearman',
            aggregation_strategy='min',
            verbose=False,
        )

    all_rows = []

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")

        adata = load_3m_result(root_dir, method)
        if adata is None:
            print(f"  SKIP: no result found")
            continue

        print(f"  Shape: {adata.shape}")
        print(f"  obsm keys: {list(adata.obsm.keys())}")
        print(f"  obs cols: {list(adata.obs.columns)[:10]}")

        embedding, emb_key = get_embedding(adata, method)
        if embedding is None:
            print(f"  SKIP: no embedding found")
            continue
        print(f"  Embedding: {emb_key} → {embedding.shape}")

        # Spatial coordinates
        spatial_coords = None
        if 'spatial' in adata.obsm:
            spatial_coords = adata.obsm['spatial']

        # Adjacency matrix
        adj_matrix = knn_adj(embedding)

        # Align cell counts between embedding and GT
        n_embed = embedding.shape[0]
        y_GT_aligned = None
        if y_GT is not None:
            min_n = min(len(y_GT), n_embed)
            y_GT_aligned = y_GT[:min_n]
            if n_embed > min_n:
                embedding_eval = embedding[:min_n]
                spatial_eval = spatial_coords[:min_n] if spatial_coords is not None else None
                adj_eval = adj_matrix[:min_n, :min_n]
            else:
                embedding_eval = embedding
                spatial_eval = spatial_coords
                adj_eval = adj_matrix
        else:
            embedding_eval = embedding
            spatial_eval = spatial_coords
            adj_eval = adj_matrix

        # --- SC + BioC metrics per clustering ---
        for clust in clustering_methods:
            if clust not in adata.obs.columns:
                print(f"    [{clust}] not found, skipping")
                continue

            y_pred_raw = adata.obs[clust]
            if hasattr(y_pred_raw, 'cat'):
                y_pred = y_pred_raw.cat.codes.values
            else:
                y_pred = np.asarray(y_pred_raw.values, dtype=int)

            if y_GT_aligned is not None:
                y_pred = y_pred[:len(y_GT_aligned)]

            try:
                metrics = eval_func(
                    embeddings=embedding_eval,
                    adj_matrix=adj_eval,
                    y_pred=y_pred,
                    y_GT=y_GT_aligned,
                    spatial_coords=spatial_eval,
                    method_name=method,
                    dataset_name='3M_Simulation',
                    slice_name='3M',
                    clustering_method=clust,
                )

                row = {
                    'Method': method,
                    'Dataset': '3M_Simulation',
                    'Clustering': clust,
                    'N_Cells': len(y_pred),
                    'Embedding_Dim': embedding.shape[1],
                }
                row.update(metrics)
                all_rows.append(row)

                # Save per-method CSV
                method_out = os.path.join(output_dir, method)
                save_func(
                    metrics_dict=metrics,
                    output_dir=method_out,
                    method_name=method,
                    dataset_name='3M_Simulation',
                    slice_name='3M',
                    clustering_method=clust,
                    has_gt=(y_GT_aligned is not None),
                )

                print(f"    [{clust}] OK")
            except Exception as e:
                print(f"    [{clust}] ERROR: {e}")

        # --- CM-GTC (modality-independent, computed once) ---
        if cmgtc_evaluator is not None:
            try:
                modalities = {}
                for mod_name in ['rna', 'adt', 'atac']:
                    if mod_name in gt_data:
                        mod_data = gt_data[mod_name]
                        min_n = min(mod_data.shape[0], n_embed)
                        modalities[mod_name] = mod_data[:min_n]

                embed_for_cmgtc = embedding[:min(n_embed, min(m.shape[0] for m in modalities.values()))]
                modalities = {k: v[:embed_for_cmgtc.shape[0]] for k, v in modalities.items()}

                score, details = cmgtc_evaluator.compute_cm_gtc(embed_for_cmgtc, modalities)

                cmgtc_row = {
                    'Method': method,
                    'Dataset': '3M_Simulation',
                    'Task': '3M',
                    'N_Cells': embed_for_cmgtc.shape[0],
                    'Embedding_Dim': embedding.shape[1],
                    'CM_GTC': score,
                    'N_Modalities': len(modalities),
                }
                if 'per_modality_scores' in details:
                    for mod_name, mod_score in details['per_modality_scores'].items():
                        cmgtc_row[f'CM_GTC_{mod_name}'] = mod_score

                # Save CM-GTC separately
                cmgtc_df = pd.DataFrame([cmgtc_row])
                cmgtc_path = os.path.join(output_dir, f'cmgtc_{method}.csv')
                cmgtc_df.to_csv(cmgtc_path, index=False)

                per_mod = details.get('per_modality_scores', {})
                parts = [f"  CM-GTC = {score:.4f}"]
                for mod_key in ['rna', 'adt', 'atac']:
                    v = per_mod.get(mod_key)
                    parts.append(f"{mod_key.upper()}={v:.4f}" if v is not None else f"{mod_key.upper()}=N/A")
                print(f"{parts[0]} ({', '.join(parts[1:])})")

            except Exception as e:
                print(f"  CM-GTC ERROR: {e}")

        if test_mode:
            break

    # Save combined results
    if all_rows:
        df = pd.DataFrame(all_rows)
        out_path = os.path.join(output_dir, '3m_evaluation_results.csv')
        df.to_csv(out_path, index=False)
        print(f"\nSaved {len(df)} results to: {out_path}")

    # Combine CM-GTC results
    cmgtc_files = sorted(Path(output_dir).glob('cmgtc_*.csv'))
    if cmgtc_files:
        cmgtc_all = pd.concat([pd.read_csv(f) for f in cmgtc_files], ignore_index=True)
        cmgtc_all.to_csv(os.path.join(output_dir, 'cmgtc_3m_combined.csv'), index=False)

    return all_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    root = os.path.abspath(args.root)

    if args.version == 'v2':
        default_methods = METHODS_3Mv2
    else:
        default_methods = METHODS_3M
    methods = args.methods or default_methods
    clustering = args.clustering or CLUSTERING_METHODS

    print(f"Root:       {root}")
    print(f"Version:    {args.version}")
    print(f"Methods:    {methods}")
    print(f"Clustering: {clustering}")
    print(f"Test mode:  {args.test}")

    total_start = time.time()
    results = evaluate_3m(root, methods, clustering, args.test, version=args.version)
    elapsed = time.time() - total_start

    print(f"\n3M evaluation complete: {len(results)} results in {elapsed/60:.1f} min")


if __name__ == '__main__':
    main()

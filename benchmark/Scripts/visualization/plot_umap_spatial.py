#!/usr/bin/env python3
"""
SMOBench Visualization Script
Generate UMAP and spatial plots from integration results

This script reads AnnData objects containing integration results and generates
visualization plots for different clustering methods. It supports both UMAP 
and spatial coordinate plotting.

Usage:
    python plot_umap_spatial.py --adata_path <path> --method <method_name> 
                                --dataset <dataset_name> --subset <subset_name>
                                [--clustering_methods method1,method2,...]
                                [--embedding_key <key>] [--plot_dir <dir>]
"""

import os
import argparse
import re
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Suppress scanpy warnings
import warnings
warnings.filterwarnings('ignore')

# Set scanpy settings
sc.settings.verbosity = 1  # Reduce verbosity
sc.settings.set_figure_params(dpi=300, facecolor='white')


def parse_dataset_info(adata_path, dataset=None, subset=None):
    """
    Extract dataset_name and subset_name from adata_path or manual specification
    
    Parameters:
        adata_path: Path to the AnnData file
        dataset: Manual dataset specification
        subset: Manual subset specification
        
    Returns:
        tuple: (dataset_name, subset_name)
    """
    if dataset and subset:
        return dataset, subset
    
    # Auto parse from adata_path
    # Pattern: Results/adata/method/dataset/subset/adata_integrated.h5ad
    match = re.search(r'Results/adata/[^/]+/([^/]+)/([^/]+)/[^/]+\.h5ad', adata_path)
    if match:
        return match.group(1), match.group(2)
    
    # Alternative pattern for different path structures
    match = re.search(r'/([^/]+)/([^/]+)/[^/]*\.h5ad$', adata_path)
    if match:
        return match.group(1), match.group(2)
    
    return "Unknown", "Unknown"


def setup_plot_directory(method_name, dataset_name, subset_name, base_dir="Results/plot"):
    """
    Create and return plot directory path
    
    Parameters:
        method_name: Name of the integration method
        dataset_name: Dataset name
        subset_name: Subset/sample name
        base_dir: Base directory for plots
        
    Returns:
        str: Full path to plot directory
    """
    plot_dir = os.path.join(base_dir, method_name, dataset_name, subset_name)
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def check_and_compute_umap(adata, embedding_key, n_neighbors=30, force_recompute=False):
    """
    Check if UMAP exists and compute if necessary
    
    Parameters:
        adata: AnnData object
        embedding_key: Key for embeddings in adata.obsm
        n_neighbors: Number of neighbors for UMAP computation
        force_recompute: Whether to force recomputation of UMAP
        
    Returns:
        bool: True if UMAP was computed/exists, False otherwise
    """
    if embedding_key not in adata.obsm:
        print(f"Warning: Embedding key '{embedding_key}' not found in adata.obsm")
        print(f"Available keys: {list(adata.obsm.keys())}")
        return False
        
    # Check if UMAP already exists
    if 'X_umap' in adata.obsm and not force_recompute:
        print("UMAP coordinates already exist, using existing coordinates")
        return True
        
    print(f"Computing UMAP using embedding '{embedding_key}'...")
    try:
        # Compute neighbors and UMAP
        sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=n_neighbors)
        sc.tl.umap(adata)
        print("UMAP computation completed")
        return True
    except Exception as e:
        print(f"Error computing UMAP: {e}")
        return False


def plot_clustering_results(adata, clustering_methods, method_name, plot_dir, 
                          spatial_flip_y=True, point_size=20):
    """
    Generate UMAP and spatial plots for different clustering methods
    
    Parameters:
        adata: AnnData object with clustering results
        clustering_methods: List of clustering method names
        method_name: Integration method name for plot titles
        plot_dir: Directory to save plots
        spatial_flip_y: Whether to flip Y coordinates for spatial plots
        point_size: Size of points in plots
    """
    print(f"Generating plots for clustering methods: {clustering_methods}")
    
    # Check available clustering results
    available_methods = []
    for method in clustering_methods:
        if method in adata.obs.columns:
            available_methods.append(method)
        else:
            print(f"Warning: Clustering result '{method}' not found in adata.obs")
    
    if not available_methods:
        print("No clustering results found for plotting")
        return
    
    # Prepare spatial coordinates if available
    has_spatial = 'spatial' in adata.obsm
    if has_spatial and spatial_flip_y:
        # Create a copy to avoid modifying original data
        spatial_coords = adata.obsm['spatial'].copy()
        spatial_coords[:, 1] = -1 * spatial_coords[:, 1]
        adata.obsm['spatial_flipped'] = spatial_coords
        spatial_basis = 'spatial_flipped'
    else:
        spatial_basis = 'spatial'
    
    # Generate plots for each clustering method
    for method in available_methods:
        print(f"Plotting {method} clustering results...")
        
        # Create subplots
        if has_spatial:
            fig, ax_list = plt.subplots(1, 2, figsize=(12, 5))
        else:
            fig, ax_list = plt.subplots(1, 1, figsize=(6, 5))
            ax_list = [ax_list]  # Make it a list for consistent handling
        
        # Plot UMAP
        sc.pl.umap(adata, color=method, ax=ax_list[0], 
                  title=f'{method_name}-{method}', s=point_size, 
                  show=False, frameon=False)
        
        # Plot spatial if available
        if has_spatial:
            sc.pl.embedding(adata, basis=spatial_basis, color=method, 
                          ax=ax_list[1], title=f'{method_name}-{method}', 
                          s=point_size, show=False, frameon=False)
        
        # Adjust layout and save
        plt.tight_layout(w_pad=0.3)
        
        # Save with appropriate filename
        if has_spatial:
            filename = f'clustering_{method}_umap_spatial.png'
        else:
            filename = f'clustering_{method}_umap.png'
            
        save_path = os.path.join(plot_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {save_path}")


def plot_embeddings_comparison(adata, embedding_keys, method_name, plot_dir, point_size=20):
    """
    Generate UMAP plots comparing different embeddings
    
    Parameters:
        adata: AnnData object
        embedding_keys: List of embedding keys to compare
        method_name: Method name for plot titles
        plot_dir: Directory to save plots
        point_size: Size of points in plots
    """
    available_embeddings = [key for key in embedding_keys if key in adata.obsm]
    
    if not available_embeddings:
        print("No valid embeddings found for comparison")
        return
    
    print(f"Generating embedding comparison plots for: {available_embeddings}")
    
    # Generate individual UMAP plots for each embedding
    for emb_key in available_embeddings:
        print(f"Computing UMAP for embedding: {emb_key}")
        
        # Temporarily store current UMAP if exists
        temp_umap = None
        if 'X_umap' in adata.obsm:
            temp_umap = adata.obsm['X_umap'].copy()
        
        # Compute UMAP for this embedding
        try:
            sc.pp.neighbors(adata, use_rep=emb_key, n_neighbors=30)
            sc.tl.umap(adata)
            
            # Plot UMAP (color by first available clustering if exists)
            color_key = None
            for method in ['mclust', 'leiden', 'louvain', 'kmeans']:
                if method in adata.obs.columns:
                    color_key = method
                    break
            
            if color_key is None:
                # No clustering available, just plot coordinates
                fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                sc.pl.umap(adata, ax=ax, title=f'{method_name}-{emb_key}', 
                          s=point_size, show=False, frameon=False)
            else:
                fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                sc.pl.umap(adata, color=color_key, ax=ax, 
                          title=f'{method_name}-{emb_key}', 
                          s=point_size, show=False, frameon=False)
            
            # Save plot
            save_path = os.path.join(plot_dir, f'embedding_{emb_key}_umap.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved embedding plot: {save_path}")
            
        except Exception as e:
            print(f"Error plotting embedding {emb_key}: {e}")
        
        # Restore original UMAP if existed
        if temp_umap is not None:
            adata.obsm['X_umap'] = temp_umap


def main(args):
    """Main function to orchestrate the visualization process"""
    
    print(f"Starting visualization for: {args.adata_path}")
    
    # Load AnnData
    try:
        adata = sc.read_h5ad(args.adata_path)
        print(f"Loaded AnnData: {adata}")
    except Exception as e:
        print(f"Error loading AnnData: {e}")
        return
    
    # Parse dataset information
    dataset_name, subset_name = parse_dataset_info(
        args.adata_path, args.dataset, args.subset
    )
    print(f"Dataset: {dataset_name}, Subset: {subset_name}")
    
    # Setup plot directory
    plot_dir = setup_plot_directory(args.method, dataset_name, subset_name, args.plot_dir)
    print(f"Plots will be saved to: {plot_dir}")
    
    # Parse clustering methods
    clustering_methods = [method.strip() for method in args.clustering_methods.split(',')]
    
    # Check and compute UMAP if needed
    umap_success = check_and_compute_umap(
        adata, args.embedding_key, 
        n_neighbors=args.n_neighbors, 
        force_recompute=args.force_recompute
    )
    
    if not umap_success:
        print("Failed to compute/load UMAP coordinates")
        return
    
    # Generate clustering plots
    if clustering_methods and clustering_methods != ['']:
        plot_clustering_results(
            adata, clustering_methods, args.method, plot_dir,
            spatial_flip_y=args.spatial_flip_y,
            point_size=args.point_size
        )
    
    # Generate embedding comparison plots if requested
    if args.compare_embeddings:
        embedding_keys = [key.strip() for key in args.compare_embeddings.split(',')]
        plot_embeddings_comparison(
            adata, embedding_keys, args.method, plot_dir,
            point_size=args.point_size
        )
    
    print("Visualization completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate UMAP and spatial plots from SMOBench integration results'
    )
    
    # Required arguments
    parser.add_argument('--adata_path', type=str, required=True,
                       help='Path to integrated AnnData (.h5ad) file')
    parser.add_argument('--method', type=str, required=True,
                       help='Integration method name (e.g., SpatialGlue, PRAGA)')
    
    # Dataset information
    parser.add_argument('--dataset', type=str, default='',
                       help='Dataset name (auto-extracted if not provided)')
    parser.add_argument('--subset', type=str, default='',
                       help='Subset/sample name (auto-extracted if not provided)')
    
    # Clustering and embedding options
    parser.add_argument('--clustering_methods', type=str, 
                       default='mclust,leiden,louvain,kmeans',
                       help='Comma-separated list of clustering methods to plot')
    parser.add_argument('--embedding_key', type=str, required=True,
                       help='Key for integration embeddings in adata.obsm')
    parser.add_argument('--compare_embeddings', type=str, default='',
                       help='Comma-separated list of embedding keys to compare')
    
    # Plot settings
    parser.add_argument('--plot_dir', type=str, default='Results/plot',
                       help='Base directory for saving plots')
    parser.add_argument('--point_size', type=float, default=20,
                       help='Size of points in plots')
    parser.add_argument('--n_neighbors', type=int, default=30,
                       help='Number of neighbors for UMAP computation')
    parser.add_argument('--spatial_flip_y', action='store_true', default=True,
                       help='Flip Y coordinates for spatial plots')
    parser.add_argument('--force_recompute', action='store_true',
                       help='Force recomputation of UMAP coordinates')
    
    args = parser.parse_args()
    main(args)
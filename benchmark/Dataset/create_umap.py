#!/usr/bin/env python3
"""
Script to create 2D UMAP visualization plots for all datasets with labels
"""
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from pathlib import Path
import os

# Set up plotting parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 300
sns.set_style("whitegrid")

# Define base data directory
DATA_DIR = "/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench_Data"
OUTPUT_DIR = "/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench_Data/_plot_umap"

def compute_umap_if_needed(adata, n_neighbors=15, min_dist=0.1):
    """Compute UMAP if not already present"""
    if 'X_umap' not in adata.obsm:
        print("Computing UMAP embeddings...")
        # Normalize and log-transform if needed
        if 'log1p' not in adata.uns:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        
        # Compute PCA if needed
        if 'X_pca' not in adata.obsm:
            sc.tl.pca(adata, n_comps=min(50, min(adata.n_obs, adata.n_vars) - 1))
        
        # Compute neighbors if needed
        if 'neighbors' not in adata.uns:
            n_pcs = min(50, adata.obsm['X_pca'].shape[1])
            sc.pp.neighbors(adata, n_neighbors=min(n_neighbors, adata.n_obs - 1), n_pcs=n_pcs)
        
        # Compute UMAP
        sc.tl.umap(adata, min_dist=min_dist)
    else:
        print("UMAP already computed")
    
    return adata

def plot_umap_with_labels(adata, title, output_path, label_col=None):
    """Create UMAP plot with labels if available"""
    
    # Compute UMAP if needed
    try:
        adata = compute_umap_if_needed(adata)
    except Exception as e:
        print(f"Error computing UMAP for {title}: {e}")
        return False
    
    # Get UMAP coordinates
    if 'X_umap' not in adata.obsm:
        print(f"No UMAP coordinates found for {title}")
        return False
    
    umap_coords = adata.obsm['X_umap']
    x, y = umap_coords[:, 0], umap_coords[:, 1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if label_col and label_col in adata.obs.columns:
        # Plot with labels (colored by label)
        labels = adata.obs[label_col].astype(str)
        unique_labels = sorted(labels.unique())
        
        # Use a colormap with enough colors
        if len(unique_labels) <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        else:
            colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(x[mask], y[mask], c=[colors[i]], 
                      label=label, alpha=0.7, s=20, edgecolors='none')
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, 
                 fontsize=8, markerscale=1.5)
        plt.title(f'{title}\nUMAP colored by {label_col} ({len(unique_labels)} clusters)', 
                 fontsize=12, fontweight='bold')
        
    else:
        # Plot without labels (single color)
        ax.scatter(x, y, c='lightblue', alpha=0.7, s=20, edgecolors='none')
        plt.title(f'{title}\nUMAP - No labels available', 
                 fontsize=12, fontweight='bold')
    
    ax.set_xlabel('UMAP1', fontsize=11)
    ax.set_ylabel('UMAP2', fontsize=11)
    ax.set_aspect('equal')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved UMAP plot: {output_path}")
    return True

def find_label_columns(adata):
    """Find potential label columns in the dataset"""
    label_keywords = ['label', 'cluster', 'celltype', 'cell_type', 'annotation', 
                     'group', 'class', 'type', 'leiden', 'louvain', 'domain']
    
    label_columns = []
    
    # Print all columns for debugging
    print(f"  Available columns in .obs: {list(adata.obs.columns)}")
    
    for col in adata.obs.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in label_keywords):
            label_columns.append(col)
    
    return label_columns

def create_all_umap_plots():
    """Create UMAP plots for all datasets with labels"""
    
    # Find all h5ad files in the data directory
    search_path = os.path.join(DATA_DIR, "**/*.h5ad")
    h5ad_files = glob.glob(search_path, recursive=True)
    h5ad_files.sort()
    
    print(f"Found {len(h5ad_files)} h5ad files in {DATA_DIR}")
    
    # Filter out unwanted directories
    exclude_patterns = ['SMOBench-CLEAN-main', 'Methods', 'Reproduce']
    h5ad_files = [f for f in h5ad_files if not any(pattern in f for pattern in exclude_patterns)]
    
    print(f"After filtering: {len(h5ad_files)} files to process")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    successful_plots = 0
    failed_plots = 0
    
    for filepath in h5ad_files:
        try:
            print(f"\n{'='*60}")
            print(f"Processing: {filepath}")
            
            # Load data
            adata = sc.read_h5ad(filepath)
            print(f"Loaded data: {adata.n_obs} cells, {adata.n_vars} genes")
            
            # Create meaningful title from relative path
            rel_path = os.path.relpath(filepath, DATA_DIR)
            path_parts = Path(rel_path).parts
            
            # Use directory structure for title
            if len(path_parts) > 1:
                dataset_name = path_parts[0]
                if len(path_parts) > 2:
                    subdataset = path_parts[-2]
                    title = f"{dataset_name}_{subdataset}"
                else:
                    title = dataset_name
            else:
                title = Path(filepath).stem
            
            # Find label columns
            label_columns = find_label_columns(adata)
            print(f"Found label columns: {label_columns}")
            
            if label_columns:
                # Create plots for each label column
                for label_col in label_columns:
                    n_unique = adata.obs[label_col].nunique()
                    print(f"  Creating UMAP for '{label_col}' ({n_unique} unique values)")
                    
                    output_filename = f"{title}_{label_col}_umap.png"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    
                    success = plot_umap_with_labels(adata, title, output_path, label_col)
                    if success:
                        successful_plots += 1
                    else:
                        failed_plots += 1
            else:
                # Create plot without labels
                print("  No label columns found, creating unlabeled UMAP")
                output_filename = f"{title}_no_labels_umap.png"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                
                success = plot_umap_with_labels(adata, title, output_path)
                if success:
                    successful_plots += 1
                else:
                    failed_plots += 1
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            import traceback
            traceback.print_exc()
            failed_plots += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully created plots: {successful_plots}")
    print(f"Failed to create plots: {failed_plots}")
    print(f"Output directory: {OUTPUT_DIR}")

def create_summary_umap_plot():
    """Create a summary figure showing all datasets in UMAP subplots"""
    
    # Find all h5ad files
    search_path = os.path.join(DATA_DIR, "**/*.h5ad")
    h5ad_files = glob.glob(search_path, recursive=True)
    
    # Filter out unwanted directories
    exclude_patterns = ['SMOBench-CLEAN-main', 'Methods', 'Reproduce']
    h5ad_files = [f for f in h5ad_files if not any(pattern in f for pattern in exclude_patterns)]
    h5ad_files.sort()
    
    # Limit to reasonable number for summary
    if len(h5ad_files) > 20:
        print(f"Found {len(h5ad_files)} files, limiting summary to first 20")
        h5ad_files = h5ad_files[:20]
    
    # Create a large summary plot
    n_files = len(h5ad_files)
    n_cols = 4
    n_rows = (n_files + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if n_files > 1 else [axes]
    
    for i, filepath in enumerate(h5ad_files):
        try:
            print(f"Adding to summary: {os.path.basename(filepath)}")
            adata = sc.read_h5ad(filepath)
            ax = axes[i]
            
            # Compute UMAP if needed
            adata = compute_umap_if_needed(adata)
            
            if 'X_umap' not in adata.obsm:
                continue
            
            # Get UMAP coordinates
            umap_coords = adata.obsm['X_umap']
            x, y = umap_coords[:, 0], umap_coords[:, 1]
            
            # Find label column
            label_columns = find_label_columns(adata)
            
            if label_columns:
                label_col = label_columns[0]  # Use first label column
                labels = adata.obs[label_col].astype(str)
                unique_labels = sorted(labels.unique())
                
                # Use colors
                if len(unique_labels) <= 20:
                    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
                else:
                    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(unique_labels)))
                
                for j, label in enumerate(unique_labels):
                    mask = labels == label
                    ax.scatter(x[mask], y[mask], c=[colors[j]], 
                              alpha=0.7, s=5, edgecolors='none')
            else:
                ax.scatter(x, y, c='lightblue', alpha=0.7, s=5, edgecolors='none')
            
            # Set title from relative path
            rel_path = os.path.relpath(filepath, DATA_DIR)
            path_parts = Path(rel_path).parts
            title = path_parts[0] if len(path_parts) > 0 else "Unknown"
            
            ax.set_title(title, fontsize=9)
            ax.set_xlabel('UMAP1', fontsize=8)
            ax.set_ylabel('UMAP2', fontsize=8)
            ax.tick_params(labelsize=7)
            
        except Exception as e:
            print(f"Error in summary plot for {filepath}: {e}")
    
    # Hide unused subplots
    for i in range(len(h5ad_files), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    summary_path = os.path.join(OUTPUT_DIR, "summary_all_datasets_umap.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created summary UMAP plot: {summary_path}")

def main():
    """Main function"""
    print("Creating UMAP visualization plots...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*60)
    
    # Create individual plots
    create_all_umap_plots()
    
    print("\n" + "="*60)
    print("Creating summary plot...")
    create_summary_umap_plot()
    
    print("\n" + "="*60)
    print("All UMAP plots completed!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script to create 2D visualization plots for all datasets with spatial coordinates and labels
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

def plot_spatial_with_labels(adata, title, output_path, label_col=None):
    """Create spatial plot with labels if available"""
    
    # Get spatial coordinates
    if 'spatial' in adata.obsm:
        coords = adata.obsm['spatial']
        x, y = coords[:, 0], coords[:, 1]
    elif 'x' in adata.obs.columns and 'y' in adata.obs.columns:
        x, y = adata.obs['x'].values, adata.obs['y'].values
    else:
        print(f"No spatial coordinates found for {title}")
        return False
    
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
                      label=label, alpha=0.8, s=50)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
        plt.title(f'{title}\nColored by {label_col} ({len(unique_labels)} clusters)')
        
    else:
        # Plot without labels (single color)
        ax.scatter(x, y, c='lightblue', alpha=0.8, s=50)
        plt.title(f'{title}\nNo labels available')
    
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_aspect('equal')
    
    # Invert y-axis to match typical spatial transcriptomics orientation
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {output_path}")
    return True

def find_label_columns(adata):
    """Find potential label columns in the dataset"""
    label_keywords = ['label', 'cluster', 'celltype', 'cell_type', 'annotation', 
                     'group', 'class', 'type', 'leiden', 'louvain']
    
    label_columns = []
    for col in adata.obs.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in label_keywords):
            label_columns.append(col)
    
    return label_columns

def create_all_spatial_plots():
    """Create spatial plots for all datasets with labels"""
    
    # Find all h5ad files
    h5ad_files = glob.glob("**/*.h5ad", recursive=True)
    h5ad_files.sort()
    
    output_base = "/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench_Data/_plot_myx"
    
    successful_plots = 0
    failed_plots = 0
    
    for filepath in h5ad_files:
        try:
            print(f"\nProcessing: {filepath}")
            
            # Load data
            adata = sc.read_h5ad(filepath)
            
            # Create meaningful title
            path_parts = Path(filepath).parts
            if len(path_parts) >= 2:
                dataset_group = path_parts[0]
                if len(path_parts) >= 3:
                    experiment = path_parts[1]
                else:
                    experiment = "main"
                filename = Path(filepath).stem
                title = f"{dataset_group}_{experiment}_{filename}"
            else:
                title = Path(filepath).stem
            
            # Find label columns
            label_columns = find_label_columns(adata)
            
            if label_columns:
                # Create plots for each label column
                for label_col in label_columns:
                    output_filename = f"{title}_{label_col}.png"
                    output_path = os.path.join(output_base, output_filename)
                    
                    success = plot_spatial_with_labels(adata, title, output_path, label_col)
                    if success:
                        successful_plots += 1
                    else:
                        failed_plots += 1
            else:
                # Create plot without labels
                output_filename = f"{title}_no_labels.png"
                output_path = os.path.join(output_base, output_filename)
                
                success = plot_spatial_with_labels(adata, title, output_path)
                if success:
                    successful_plots += 1
                else:
                    failed_plots += 1
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            failed_plots += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully created plots: {successful_plots}")
    print(f"Failed to create plots: {failed_plots}")
    print(f"Output directory: {output_base}")

def create_summary_plot():
    """Create a summary figure showing all datasets in subplots"""
    
    # Find all h5ad files with labels
    h5ad_files = glob.glob("**/*.h5ad", recursive=True)
    h5ad_files = [f for f in h5ad_files if "Mouse_Thymus" not in f and "Mouse_Spleen" not in f]  # Skip files without labels
    h5ad_files.sort()
    
    # Create a large summary plot
    n_files = len(h5ad_files)
    n_cols = 4
    n_rows = (n_files + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if n_files > 1 else [axes]
    
    for i, filepath in enumerate(h5ad_files):
        try:
            adata = sc.read_h5ad(filepath)
            ax = axes[i]
            
            # Get spatial coordinates
            if 'spatial' in adata.obsm:
                coords = adata.obsm['spatial']
                x, y = coords[:, 0], coords[:, 1]
            elif 'x' in adata.obs.columns and 'y' in adata.obs.columns:
                x, y = adata.obs['x'].values, adata.obs['y'].values
            else:
                continue
            
            # Find label column
            label_columns = find_label_columns(adata)
            
            if label_columns:
                label_col = label_columns[0]  # Use first label column
                labels = adata.obs[label_col].astype(str)
                unique_labels = sorted(labels.unique())
                
                # Use colors
                colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
                
                for j, label in enumerate(unique_labels):
                    mask = labels == label
                    ax.scatter(x[mask], y[mask], c=[colors[j]], 
                              alpha=0.8, s=15)
            else:
                ax.scatter(x, y, c='lightblue', alpha=0.8, s=15)
            
            # Set title
            path_parts = Path(filepath).parts
            title = f"{path_parts[0]}"
            if len(path_parts) >= 3:
                title += f"_{path_parts[1]}"
            ax.set_title(title, fontsize=10)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            
        except Exception as e:
            print(f"Error in summary plot for {filepath}: {e}")
    
    # Hide unused subplots
    for i in range(len(h5ad_files), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    summary_path = "/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench_Data/_plot_myx/summary_all_datasets.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created summary plot: {summary_path}")

def main():
    """Main function"""
    print("Creating spatial visualization plots...")
    
    # Create individual plots
    create_all_spatial_plots()
    
    print("\nCreating summary plot...")
    create_summary_plot()
    
    print("\nAll plots completed!")

if __name__ == "__main__":
    main()
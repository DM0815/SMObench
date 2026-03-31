#!/usr/bin/env python3
"""
SMOBench Batch Visualization Script
Generate plots for multiple integration results automatically

This script processes multiple AnnData files and generates visualization plots
for all available integration methods and clustering results.

Usage:
    python batch_visualization.py --results_dir <dir> [options]
    python batch_visualization.py --file_list <file> [options]
"""

import os
import glob
import argparse
import subprocess
import pandas as pd
import scanpy as sc
from pathlib import Path
import json


def find_adata_files(results_dir, pattern="**/*.h5ad"):
    """
    Find all AnnData files in results directory
    
    Parameters:
        results_dir: Base results directory
        pattern: Glob pattern for finding files
        
    Returns:
        list: List of AnnData file paths
    """
    search_path = os.path.join(results_dir, pattern)
    files = glob.glob(search_path, recursive=True)
    return sorted(files)


def detect_method_from_path(file_path):
    """
    Detect integration method from file path
    
    Parameters:
        file_path: Path to AnnData file
        
    Returns:
        str: Method name or 'Unknown'
    """
    # Common method patterns
    methods = ['SpatialGlue', 'PRAGA', 'SpaMosaic', 'COSMOS', 'PRESENT', 'SpaMultiVAE']
    
    for method in methods:
        if method in file_path:
            return method
    
    return 'Unknown'


def detect_embedding_key(adata_path, method_name):
    """
    Detect the main embedding key from AnnData file
    
    Parameters:
        adata_path: Path to AnnData file
        method_name: Integration method name
        
    Returns:
        str: Embedding key or None if not found
    """
    try:
        adata = sc.read_h5ad(adata_path)
        
        # Method-specific embedding keys
        method_keys = {
            'SpatialGlue': ['SpatialGlue', 'spatial_emb'],
            'PRAGA': ['PRAGA', 'PRAGA_emb'],
            'SpaMosaic': ['merged_emb', 'SpaMosaic'],
            'COSMOS': ['COSMOS', 'cosmos_emb'],
            'PRESENT': ['PRESENT', 'present_emb'],
            'SpaMultiVAE': ['SpaMultiVAE', 'spa_emb']
        }
        
        # Check method-specific keys first
        if method_name in method_keys:
            for key in method_keys[method_name]:
                if key in adata.obsm:
                    return key
        
        # Check for common embedding keys
        common_keys = ['spatial_emb', 'merged_emb', 'integrated_emb', 'latent', 'emb']
        for key in common_keys:
            if key in adata.obsm:
                return key
        
        # Return first available embedding key
        available_keys = list(adata.obsm.keys())
        if available_keys:
            return available_keys[0]
        
        return None
        
    except Exception as e:
        print(f"Error reading {adata_path}: {e}")
        return None


def detect_clustering_methods(adata_path):
    """
    Detect available clustering methods in AnnData file
    
    Parameters:
        adata_path: Path to AnnData file
        
    Returns:
        list: Available clustering method names
    """
    try:
        adata = sc.read_h5ad(adata_path)
        
        # Common clustering method names
        clustering_names = ['mclust', 'leiden', 'louvain', 'kmeans', 'spectral', 'hierarchical']
        
        available = []
        for method in clustering_names:
            if method in adata.obs.columns:
                available.append(method)
        
        return available
        
    except Exception as e:
        print(f"Error reading {adata_path}: {e}")
        return []


def run_visualization(adata_path, method_name, embedding_key, clustering_methods=None, 
                     output_dir="Results/plot", **kwargs):
    """
    Run visualization script for a single AnnData file
    
    Parameters:
        adata_path: Path to AnnData file
        method_name: Integration method name
        embedding_key: Embedding key in adata.obsm
        clustering_methods: List of clustering methods
        output_dir: Output directory for plots
        **kwargs: Additional arguments for visualization script
        
    Returns:
        bool: True if successful, False otherwise
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    viz_script = os.path.join(script_dir, "plot_umap_spatial.py")
    
    if not os.path.exists(viz_script):
        print(f"Error: Visualization script not found: {viz_script}")
        return False
    
    # Build command
    cmd = [
        "conda", "run", "-n", "smobench", "python", viz_script,
        "--adata_path", adata_path,
        "--method", method_name,
        "--embedding_key", embedding_key,
        "--plot_dir", output_dir
    ]
    
    # Add clustering methods if specified
    if clustering_methods:
        cmd.extend(["--clustering_methods", ",".join(clustering_methods)])
    
    # Add additional arguments
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key.replace('_', '-')}")
        else:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    print(f"Running visualization for: {os.path.basename(adata_path)}")
    print(f"Method: {method_name}, Embedding: {embedding_key}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("Visualization completed successfully")
            return True
        else:
            print(f"Visualization failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("Visualization timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"Error running visualization: {e}")
        return False


def create_visualization_summary(results, output_file):
    """
    Create a summary report of visualization results
    
    Parameters:
        results: List of result dictionaries
        output_file: Path to save summary
    """
    summary_data = []
    
    for result in results:
        summary_data.append({
            'file_path': result['file_path'],
            'method': result['method'],
            'embedding_key': result['embedding_key'],
            'clustering_methods': ','.join(result.get('clustering_methods', [])),
            'success': result['success'],
            'error': result.get('error', '')
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False)
    
    # Print summary statistics
    total = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total - successful
    
    print(f"\n{'='*60}")
    print("BATCH VISUALIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/total*100:.1f}%")
    print(f"Summary saved to: {output_file}")
    print(f"{'='*60}")


def main(args):
    """Main function for batch visualization"""
    
    print("Starting batch visualization...")
    
    # Get list of files to process
    if args.file_list:
        # Read from file list
        with open(args.file_list, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
    elif args.results_dir:
        # Find files in directory
        files = find_adata_files(args.results_dir, args.pattern)
    else:
        print("Error: Either --results_dir or --file_list must be provided")
        return
    
    if not files:
        print("No AnnData files found to process")
        return
    
    print(f"Found {len(files)} files to process")
    
    # Process each file
    results = []
    
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {file_path}")
        
        # Skip if file doesn't exist
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            results.append({
                'file_path': file_path,
                'method': 'Unknown',
                'embedding_key': None,
                'success': False,
                'error': 'File not found'
            })
            continue
        
        # Detect method and embedding key
        method_name = args.method or detect_method_from_path(file_path)
        embedding_key = args.embedding_key or detect_embedding_key(file_path, method_name)
        
        if not embedding_key:
            print(f"No suitable embedding key found for: {file_path}")
            results.append({
                'file_path': file_path,
                'method': method_name,
                'embedding_key': None,
                'success': False,
                'error': 'No embedding key found'
            })
            continue
        
        # Detect clustering methods if not specified
        clustering_methods = None
        if args.clustering_methods:
            clustering_methods = [m.strip() for m in args.clustering_methods.split(',')]
        elif args.auto_detect_clustering:
            clustering_methods = detect_clustering_methods(file_path)
        
        # Prepare kwargs for visualization
        viz_kwargs = {
            'point_size': args.point_size,
            'n_neighbors': args.n_neighbors,
            'force_recompute': args.force_recompute
        }
        
        if args.no_spatial_flip:
            viz_kwargs['spatial_flip_y'] = False
        
        # Run visualization
        success = run_visualization(
            file_path, method_name, embedding_key,
            clustering_methods=clustering_methods,
            output_dir=args.output_dir,
            **viz_kwargs
        )
        
        results.append({
            'file_path': file_path,
            'method': method_name,
            'embedding_key': embedding_key,
            'clustering_methods': clustering_methods or [],
            'success': success
        })
        
        if args.stop_on_error and not success:
            print("Stopping due to error (--stop-on-error)")
            break
    
    # Create summary report
    if args.summary_file:
        create_visualization_summary(results, args.summary_file)
    
    print("\nBatch visualization completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch visualization for SMOBench integration results'
    )
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--results_dir', type=str,
                      help='Directory containing AnnData files')
    group.add_argument('--file_list', type=str,
                      help='Text file with list of AnnData file paths')
    
    # File search options
    parser.add_argument('--pattern', type=str, default='**/*.h5ad',
                       help='Glob pattern for finding files (default: **/*.h5ad)')
    
    # Method and embedding options
    parser.add_argument('--method', type=str,
                       help='Force specific method name (auto-detected if not provided)')
    parser.add_argument('--embedding_key', type=str,
                       help='Force specific embedding key (auto-detected if not provided)')
    parser.add_argument('--clustering_methods', type=str,
                       help='Comma-separated clustering methods')
    parser.add_argument('--auto_detect_clustering', action='store_true',
                       help='Auto-detect available clustering methods')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='Results/plot',
                       help='Output directory for plots (default: Results/plot)')
    parser.add_argument('--summary_file', type=str, default='batch_visualization_summary.csv',
                       help='Summary report file (default: batch_visualization_summary.csv)')
    
    # Plot options
    parser.add_argument('--point_size', type=float, default=20,
                       help='Point size for plots (default: 20)')
    parser.add_argument('--n_neighbors', type=int, default=30,
                       help='Number of neighbors for UMAP (default: 30)')
    parser.add_argument('--no_spatial_flip', action='store_true',
                       help="Don't flip Y coordinates for spatial plots")
    parser.add_argument('--force_recompute', action='store_true',
                       help='Force recomputation of UMAP coordinates')
    
    # Execution options
    parser.add_argument('--stop_on_error', action='store_true',
                       help='Stop processing on first error')
    
    args = parser.parse_args()
    main(args)
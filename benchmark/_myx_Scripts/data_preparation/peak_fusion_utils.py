#!/usr/bin/env python3
"""
Optimized peak fusion algorithms for ATAC-seq data
Uses interval trees and hash tables for improved time complexity
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from collections import defaultdict
import re

def parse_peak_coordinates(peak_name):
    """Parse peak coordinates: chr1:1000-2000 or chr1-1000-2000"""
    try:
        # Try colon format: chr1:1000-2000
        match = re.match(r'(.+):(\d+)-(\d+)', peak_name)
        if match:
            chrom, start, end = match.groups()
            return chrom, int(start), int(end)
        
        # Try dash format: chr1-1000-2000
        match = re.match(r'(.+)-(\d+)-(\d+)', peak_name)
        if match:
            chrom, start, end = match.groups()
            return chrom, int(start), int(end)
    except:
        pass
    return None, None, None

def create_optimized_peak_mapping(all_peak_lists, overlap_threshold=0.5):
    """
    Optimized peak merging algorithm
    Time complexity: O(n log n) instead of O(n²)
    """
    print("    Parsing peak coordinates...")
    
    # Collect all peaks and parse coordinates
    all_peaks_info = []
    for dataset_idx, peaks in enumerate(all_peak_lists):
        for peak_idx, peak in enumerate(peaks):
            chrom, start, end = parse_peak_coordinates(peak)
            if chrom is not None:
                all_peaks_info.append({
                    'name': peak,
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'dataset': dataset_idx,
                    'original_idx': peak_idx
                })
    
    print(f"    Successfully parsed {len(all_peaks_info)}/{sum(len(p) for p in all_peak_lists)} peaks")
    
    if len(all_peaks_info) == 0:
        return {}
    
    # Group by chromosome for processing
    print("    Grouping by chromosome...")
    chrom_groups = defaultdict(list)
    for peak_info in all_peaks_info:
        chrom_groups[peak_info['chrom']].append(peak_info)
    
    peak_mapping = {}
    merged_peak_counter = 0
    
    # Process each chromosome separately
    for chrom, peaks_in_chrom in chrom_groups.items():
        print(f"    Processing chromosome {chrom}: {len(peaks_in_chrom)} peaks")
        
        # Sort by start position
        peaks_in_chrom.sort(key=lambda x: x['start'])
        
        # Use sliding window to find overlaps
        used = [False] * len(peaks_in_chrom)
        
        for i, peak1 in enumerate(peaks_in_chrom):
            if used[i]:
                continue
                
            # Create new merged peak
            merged_name = f"{chrom}:{peak1['start']}-{peak1['end']}_merged_{merged_peak_counter}"
            merged_peak_counter += 1
            
            # Add current peak to mapping
            peak_mapping[peak1['name']] = merged_name
            used[i] = True
            
            # Find subsequent overlapping peaks
            for j in range(i + 1, len(peaks_in_chrom)):
                if used[j]:
                    continue
                    
                peak2 = peaks_in_chrom[j]
                
                # If start position too far, no subsequent overlaps
                if peak2['start'] > peak1['end']:
                    break
                
                # Calculate overlap
                overlap_start = max(peak1['start'], peak2['start'])
                overlap_end = min(peak1['end'], peak2['end'])
                
                if overlap_start < overlap_end:
                    overlap_len = overlap_end - overlap_start
                    peak1_len = peak1['end'] - peak1['start']
                    peak2_len = peak2['end'] - peak2['start']
                    
                    # Calculate overlap ratio
                    overlap_ratio = overlap_len / min(peak1_len, peak2_len)
                    
                    if overlap_ratio >= overlap_threshold:
                        peak_mapping[peak2['name']] = merged_name
                        used[j] = True
    
    # Handle peaks with unparseable coordinates (direct mapping)
    for peaks in all_peak_lists:
        for peak in peaks:
            if peak not in peak_mapping:
                peak_mapping[peak] = peak
    
    print(f"    Mapping completed: {len(peak_mapping)} -> {len(set(peak_mapping.values()))} peaks")
    return peak_mapping

def aggregate_peak_data_optimized(data_matrix, original_peaks, peak_mapping):
    """
    Optimized data aggregation
    Uses dictionary grouping to avoid repeated calculations
    """
    # Create mapping dictionary: merged_peak -> [original_indices]
    merged_to_original = defaultdict(list)
    for i, original_peak in enumerate(original_peaks):
        merged_peak = peak_mapping.get(original_peak, original_peak)
        merged_to_original[merged_peak].append(i)
    
    # Create ordered list of new feature names
    merged_peaks = sorted(merged_to_original.keys())
    
    # Aggregate data
    aggregated_data = []
    for merged_peak in merged_peaks:
        original_indices = merged_to_original[merged_peak]
        
        if len(original_indices) == 1:
            # Single peak, direct copy
            aggregated_data.append(data_matrix[:, original_indices[0]])
        else:
            # Multiple peaks, sum aggregation
            peak_data = data_matrix[:, original_indices].sum(axis=1)
            aggregated_data.append(peak_data)
    
    # Combine into matrix
    if len(aggregated_data) > 0:
        new_matrix = np.column_stack(aggregated_data)
    else:
        new_matrix = np.empty((data_matrix.shape[0], 0))
    
    return new_matrix, merged_peaks

def fuse_atac_data_optimized(adatas, dataset_names):
    """ATAC data fusion using optimized algorithm"""
    print(f"  Fusing ATAC data using optimized algorithm...")
    
    # Collect all peak names
    all_peak_lists = [list(adata.var_names) for adata in adatas]
    
    # Create optimized peak mapping
    peak_mapping = create_optimized_peak_mapping(all_peak_lists)
    print(f"    Original peak total: {sum(len(peaks) for peaks in all_peak_lists)}")
    
    # Get all merged peak names (unified peak set)
    all_merged_peaks = sorted(set(peak_mapping.values()))
    print(f"    Merged peak count: {len(all_merged_peaks)}")
    
    # Build merged peak index for fast lookup
    merged_peak_to_idx = {p: idx for idx, p in enumerate(all_merged_peaks)}
    n_merged = len(all_merged_peaks)

    # Apply peak mapping to each dataset and aggregate to unified peak set (sparse)
    aggregated_adatas = []
    for i, adata in enumerate(adatas):
        print(f"    Processing dataset {i+1}/{len(adatas)}: {dataset_names[i]}")

        # Keep data as sparse
        data_matrix = sp.csc_matrix(adata.X) if not sp.issparse(adata.X) else adata.X.tocsc()
        original_peaks = list(adata.var_names)
        n_spots = data_matrix.shape[0]

        # Build sparse remapping matrix (n_original x n_merged) to aggregate peaks
        row_indices, col_indices = [], []
        for j, original_peak in enumerate(original_peaks):
            merged_peak = peak_mapping.get(original_peak, original_peak)
            if merged_peak in merged_peak_to_idx:
                row_indices.append(j)
                col_indices.append(merged_peak_to_idx[merged_peak])

        remap = sp.csr_matrix(
            (np.ones(len(row_indices), dtype=np.float32), (row_indices, col_indices)),
            shape=(len(original_peaks), n_merged)
        )

        # Sparse matrix multiply: (n_spots x n_original) @ (n_original x n_merged)
        new_matrix = (data_matrix @ remap).tocsr()

        new_adata = sc.AnnData(
            X=new_matrix,
            obs=adata.obs.copy(),
            var=pd.DataFrame(index=all_merged_peaks)
        )

        aggregated_adatas.append(new_adata)
        print(f"      {adata.shape[1]} -> {new_adata.shape[1]} peaks")
    
    # Final merge after all datasets have identical peak set
    try:
        X_list, obs_list = [], []
        for i, ad in enumerate(aggregated_adatas):
            # stabilize IDs
            ad = ad.copy()
            ad.obs.index = [f"{dataset_names[i]}:{x}" for x in ad.obs_names]
            X_list.append(ad.X)
            obs_df = ad.obs.copy()
            obs_df["batch"] = dataset_names[i]
            obs_list.append(obs_df)
            aggregated_adatas[i] = ad

        merged_X = sp.vstack(X_list).tocsr() if hasattr(X_list[0], "tocsr") else np.vstack(X_list)
        merged_obs = pd.concat(obs_list, axis=0)
        merged_var = aggregated_adatas[0].var.copy()

        merged_adata = sc.AnnData(X=merged_X, obs=merged_obs, var=merged_var)
        merged_adata.obs_names_make_unique()
        merged_adata.var_names_make_unique()

        print(f"    Final merge: {merged_adata.n_obs} cells, {merged_adata.n_vars} peaks")
        return merged_adata
    except Exception as e:
        print(f"    Merge failed: {e}")
        return None

#!/usr/bin/env python3
"""
Comprehensive AnnData Structure Checker for SMOBench Integration Results
Updated for standardized Results structure
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SMOBenchStructureChecker:
    """Comprehensive structure checker for standardized SMOBench integration methods"""
    
    def __init__(self, base_dir="/home/zhenghong/SMOBench-CLEAN"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "Results" / "adata"
        
        # Define method-specific dataset support (based on user specifications)
        self.method_support = {
            "SpatialGlue": {
                "vertical": ["HLN", "HT", "MISAR_S1", "MISAR_S2", "Mouse_Thymus", "Mouse_Spleen", "Mouse_Brain"],
                "horizontal": ["HLN", "HT", "MISAR_S1", "MISAR_S2", "Mouse_Thymus", "Mouse_Spleen", "Mouse_Brain"],
                "data_types": ["RNA_ADT", "RNA_ATAC"]
            },
            "SpaMV": {
                "vertical": ["HLN", "HT", "MISAR_S1", "MISAR_S2", "Mouse_Thymus", "Mouse_Spleen", "Mouse_Brain"],
                "horizontal": ["HLN", "HT", "MISAR_S1", "MISAR_S2", "Mouse_Thymus", "Mouse_Spleen"],
                "data_types": ["RNA_ADT", "RNA_ATAC"]
            },
            "CANDIES": {
                "vertical": ["HLN", "HT", "MISAR_S1", "MISAR_S2", "Mouse_Thymus", "Mouse_Spleen", "Mouse_Brain"],
                "horizontal": ["HLN", "HT", "MISAR_S1", "MISAR_S2", "Mouse_Spleen"],
                "data_types": ["RNA_ADT", "RNA_ATAC"]
            },
            "SpaMosaic": {
                "vertical": ["HLN", "HT", "MISAR_S1", "MISAR_S2", "Mouse_Thymus", "Mouse_Spleen", "Mouse_Brain"],
                "horizontal": ["HLN", "HT", "MISAR_S1", "MISAR_S2", "Mouse_Thymus", "Mouse_Spleen", "Mouse_Brain"],
                "data_types": ["RNA_ADT", "RNA_ATAC"]
            },
            "PRAGA": {
                "vertical": ["HLN", "HT", "MISAR_S1", "MISAR_S2", "Mouse_Thymus", "Mouse_Spleen", "Mouse_Brain"],
                "horizontal": ["HLN", "HT", "Mouse_Spleen"],
                "data_types": ["RNA_ADT"]
            },
            "PRESENT": {
                "vertical": ["HLN", "HT", "MISAR_S1", "MISAR_S2", "Mouse_Thymus", "Mouse_Spleen", "Mouse_Brain"],
                "horizontal": ["HLN", "HT", "MISAR_S1", "MISAR_S2", "Mouse_Thymus", "Mouse_Spleen", "Mouse_Brain"],
                "data_types": ["RNA_ADT", "RNA_ATAC"]
            },
            "COSMOS": {
                "vertical": ["HLN", "HT", "MISAR_S1", "MISAR_S2", "Mouse_Thymus", "Mouse_Spleen", "Mouse_Brain"],
                "horizontal": ["HLN", "HT", "MISAR_S1", "MISAR_S2", "Mouse_Thymus", "Mouse_Spleen", "Mouse_Brain"],
                "data_types": ["RNA_ADT", "RNA_ATAC"]
            },
            "SpaMultiVAE": {
                "vertical": ["HLN", "HT", "Mouse_Thymus", "Mouse_Spleen"],
                "horizontal": ["HLN", "HT", "Mouse_Thymus", "Mouse_Spleen"],
                "data_types": ["RNA_ADT"]
            }
        }
        
        # Expected clustering methods
        self.clustering_methods = ["mclust", "louvain", "leiden", "kmeans"]
        
        # Dataset slice configurations  
        self.dataset_slices = {
            "HLN": ["A1", "D1"],
            "HT": ["S1", "S2", "S3"],
            "MISAR_S1": ["E11", "E13", "E15", "E18"],
            "MISAR_S2": ["E11", "E13", "E15", "E18"],
            "Mouse_Thymus": ["Thymus1", "Thymus2", "Thymus3", "Thymus4"],
            "Mouse_Spleen": ["Spleen1", "Spleen2"],
            "Mouse_Brain": ["ATAC", "H3K27ac", "H3K27me3", "H3K4me3"]
        }
        
        # Expected cluster numbers
        self.expected_clusters = {
            "HLN": 10, "HT": 5, "MISAR_S1": 12, "MISAR_S2": 14,
            "Mouse_Thymus": 8, "Mouse_Spleen": 5, "Mouse_Brain": 18
        }
    
    def get_standardized_file_path(self, method_name, dataset, slice_name, integration_type):
        """Generate standardized file path using new naming convention"""
        
        if integration_type == "vertical":
            # Vertical: Results/adata/vertical_integration/{Method}/{Dataset}/{Slice}/{Method}_{Dataset}_{Slice}.h5ad
            filename = f"{method_name}_{dataset}_{slice_name}.h5ad"
            file_path = self.results_dir / "vertical_integration" / method_name / dataset / slice_name / filename
        else:
            # Horizontal: Results/adata/horizontal_integration/{Method}/{Dataset}/{Method}_{Dataset}_horizontal.h5ad
            filename = f"{method_name}_{dataset}_horizontal.h5ad"
            file_path = self.results_dir / "horizontal_integration" / method_name / dataset / filename
        
        return filename, file_path
    
    def validate_adata_structure(self, file_path, method_name, dataset_name):
        """Validate individual AnnData file structure"""
        result = {
            "exists": False,
            "readable": False,
            "shape": None,
            "embeddings": {},
            "clustering": {},
            "spatial": False,
            "umap": False,
            "issues": []
        }
        
        if not file_path.exists():
            result["issues"].append("File not found")
            return result
        
        result["exists"] = True
        
        try:
            with h5py.File(file_path, 'r') as f:
                result["readable"] = True
                
                # Check basic structure
                if 'X' in f:
                    result["shape"] = f['X'].shape
                
                # Check embeddings in obsm
                if 'obsm' in f:
                    obsm_keys = list(f['obsm'].keys())
                    
                    # Look for method-specific embeddings
                    method_embeddings = [k for k in obsm_keys if method_name.lower() in k.lower()]
                    if method_embeddings:
                        for emb_key in method_embeddings:
                            if emb_key in f['obsm']:
                                result["embeddings"][emb_key] = f['obsm'][emb_key].shape
                    
                    # Check for standard embeddings
                    standard_embeddings = ["X_umap", "spatial"]
                    for emb in standard_embeddings:
                        if emb in obsm_keys:
                            result["embeddings"][emb] = f['obsm'][emb].shape
                            if emb == "X_umap":
                                result["umap"] = True
                            elif emb == "spatial":
                                result["spatial"] = True
                
                # Check clustering in obs
                if 'obs' in f:
                    obs_keys = list(f['obs'].keys())
                    
                    for cluster_method in self.clustering_methods:
                        if cluster_method in obs_keys:
                            try:
                                cluster_group = f['obs'][cluster_method]
                                if isinstance(cluster_group, h5py.Group) and 'categories' in cluster_group:
                                    # Categorical data
                                    categories = cluster_group['categories'][:]
                                    n_clusters = len(categories)
                                    result["clustering"][cluster_method] = {
                                        "n_clusters": n_clusters,
                                        "type": "categorical"
                                    }
                                elif isinstance(cluster_group, h5py.Dataset):
                                    # Direct array data
                                    cluster_data = cluster_group[:]
                                    n_clusters = len(np.unique(cluster_data))
                                    result["clustering"][cluster_method] = {
                                        "n_clusters": n_clusters,
                                        "type": "array"
                                    }
                            except Exception as e:
                                result["clustering"][cluster_method] = {
                                    "n_clusters": 0,
                                    "type": "error",
                                    "error": str(e)
                                }
                
                # Validate cluster numbers
                expected_n = self.expected_clusters.get(dataset_name, 0)
                if expected_n > 0:
                    for cluster_method, cluster_info in result["clustering"].items():
                        if isinstance(cluster_info, dict) and "n_clusters" in cluster_info:
                            actual_n = cluster_info["n_clusters"]
                            if abs(actual_n - expected_n) > 3:  # Allow some tolerance
                                result["issues"].append(f"{cluster_method}: {actual_n} clusters (expected ~{expected_n})")
                
        except Exception as e:
            result["issues"].append(f"Error reading file: {str(e)}")
        
        return result
    
    def check_method_files(self, method_name):
        """Check all files for a specific method using standardized paths"""
        print(f"\n{'='*60}")
        print(f"Checking {method_name}")
        print(f"{'='*60}")
        
        method_config = self.method_support.get(method_name, {})
        total_expected = 0
        total_found = 0
        total_valid = 0
        
        results = {"vertical": {}, "horizontal": {}}
        
        # Check vertical integration
        print(f"\nVertical Integration:")
        
        for dataset in method_config.get("vertical", []):
            print(f"\n  Dataset: {dataset}")
            dataset_results = {}
            
            for slice_name in self.dataset_slices.get(dataset, []):
                total_expected += 1
                
                # Generate standardized file path
                filename, file_path = self.get_standardized_file_path(method_name, dataset, slice_name, "vertical")
                
                result = self.validate_adata_structure(file_path, method_name, dataset)
                dataset_results[slice_name] = result
                
                if result["exists"]:
                    total_found += 1
                    
                if result["readable"] and len(result["clustering"]) >= 3:  # At least 3 clustering methods
                    total_valid += 1
                    status = "✓"
                else:
                    status = "✗"
                
                print(f"    {slice_name}: {status} {result['shape'] if result['shape'] else 'N/A'}")
                if result["issues"]:
                    for issue in result["issues"]:
                        print(f"      Issue: {issue}")
                
                # Print embedding and clustering info for valid files
                if result["readable"]:
                    if result["embeddings"]:
                        print(f"      Embeddings: {list(result['embeddings'].keys())}")
                    if result["clustering"]:
                        clustering_summary = []
                        for method, info in result["clustering"].items():
                            if isinstance(info, dict) and "n_clusters" in info:
                                clustering_summary.append(f"{method}({info['n_clusters']})")
                        if clustering_summary:
                            print(f"      Clustering: {', '.join(clustering_summary)}")
            
            results["vertical"][dataset] = dataset_results
        
        # Check horizontal integration
        print(f"\nHorizontal Integration:")
        
        for dataset in method_config.get("horizontal", []):
            total_expected += 1
            
            filename, file_path = self.get_standardized_file_path(method_name, dataset, None, "horizontal")
            
            result = self.validate_adata_structure(file_path, method_name, dataset)
            results["horizontal"][dataset] = result
            
            if result["exists"]:
                total_found += 1
                
            if result["readable"] and len(result["clustering"]) >= 3:
                total_valid += 1
                status = "✓"
            else:
                status = "✗"
            
            print(f"  {dataset}: {status} {result['shape'] if result['shape'] else 'N/A'}")
            if result["issues"]:
                for issue in result["issues"]:
                    print(f"    Issue: {issue}")
            
            # Print embedding and clustering info for valid files
            if result["readable"]:
                if result["embeddings"]:
                    print(f"    Embeddings: {list(result['embeddings'].keys())}")
                if result["clustering"]:
                    clustering_summary = []
                    for cluster_method, info in result["clustering"].items():
                        if isinstance(info, dict) and "n_clusters" in info:
                            clustering_summary.append(f"{cluster_method}({info['n_clusters']})")
                    if clustering_summary:
                        print(f"    Clustering: {', '.join(clustering_summary)}")
        
        # Summary for this method
        print(f"\n{method_name} Summary:")
        print(f"  Expected files: {total_expected}")
        print(f"  Found files: {total_found}")
        print(f"  Valid files: {total_valid}")
        print(f"  Success rate: {total_valid/total_expected*100:.1f}%")
        
        return results, total_expected, total_found, total_valid
    
    def run_comprehensive_check(self):
        """Run comprehensive check for all methods using standardized structure"""
        print("SMOBench Comprehensive AnnData Structure Check (Standardized)")
        print("="*80)
        print("Checking all integration methods with standardized file structure")
        
        overall_stats = {}
        grand_total_expected = 0
        grand_total_found = 0
        grand_total_valid = 0
        
        # Check each method
        for method_name in self.method_support.keys():
            method_results, expected, found, valid = self.check_method_files(method_name)
            
            overall_stats[method_name] = {
                "expected": expected,
                "found": found,
                "valid": valid,
                "success_rate": valid/expected*100 if expected > 0 else 0
            }
            
            grand_total_expected += expected
            grand_total_found += found
            grand_total_valid += valid
        
        # Overall summary
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY")
        print(f"{'='*80}")
        
        print(f"\nMethod-wise Results:")
        print(f"{'Method':<15} {'Expected':<10} {'Found':<8} {'Valid':<8} {'Success Rate':<12}")
        print("-" * 70)
        
        for method, stats in overall_stats.items():
            print(f"{method:<15} {stats['expected']:<10} {stats['found']:<8} {stats['valid']:<8} {stats['success_rate']:<11.1f}%")
        
        print("-" * 70)
        print(f"{'TOTAL':<15} {grand_total_expected:<10} {grand_total_found:<8} {grand_total_valid:<8} {grand_total_valid/grand_total_expected*100:<11.1f}%")
        
        # Method support summary
        print(f"\nMethod Dataset Support (Standardized Structure):")
        for method, config in self.method_support.items():
            vertical_count = len(config.get("vertical", []))
            horizontal_count = len(config.get("horizontal", []))
            data_types = ", ".join(config.get("data_types", []))
            
            # Calculate total files expected
            total_vertical = sum(len(self.dataset_slices.get(d, [])) for d in config.get("vertical", []))
            total_horizontal = horizontal_count
            total_expected = total_vertical + total_horizontal
            
            print(f"  {method}: {vertical_count}V + {horizontal_count}H datasets ({data_types}) - {total_expected} files expected")
        
        # Issues summary
        print(f"\nRequired Actions:")
        if grand_total_valid < grand_total_expected:
            missing = grand_total_expected - grand_total_valid
            print(f"  - {missing} files need to be generated or fixed")
            print(f"  - Focus on methods with low success rates")
            print(f"  - Ensure all 4 clustering methods (mclust, louvain, leiden, kmeans) are present")
            print(f"  - Verify method-specific embeddings are correctly stored in obsm")
        else:
            print(f"  - All integration results are complete and valid!")
        
        print(f"\nStandardized Structure Benefits:")
        print(f"  - Consistent naming: {{Method}}_{{Dataset}}_{{Slice}}.h5ad")
        print(f"  - Clean directory structure without redundant subdirectories")
        print(f"  - Simplified evaluation script development")
        print(f"  - Easy batch processing and method comparison")
        
        return overall_stats

def main():
    """Main execution function"""
    checker = SMOBenchStructureChecker()
    results = checker.run_comprehensive_check()
    return results

if __name__ == "__main__":
    main()
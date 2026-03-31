#!/usr/bin/env python3
"""
Check generated fusion files
"""

import h5py
from pathlib import Path

def check_file_shape(file_path):
    """Get h5ad file shape"""
    try:
        import scanpy as sc
        ad = sc.read_h5ad(file_path, backed=None)
        return (ad.n_obs, ad.n_vars)
    except Exception:
        pass
    try:
        with h5py.File(file_path, 'r') as f:
            if 'X' in f:
                dset = f['X']
                shape = getattr(dset, 'shape', None)
                if shape is not None and len(shape) == 2:
                    return shape
            if 'X/indptr' in f and 'var/_index' in f:
                n_obs = f['X/indptr'].shape[0] - 1
                n_vars = f['var/_index'].shape[0]
                return (n_obs, n_vars)
            if 'X/data' in f and 'X/indptr' in f:
                n_obs = f['X/indptr'].shape[0] - 1
                n_vars = f['X/data'].shape[0]
                return (n_obs, n_vars)
    except Exception:
        pass
    return None

def main():
    base_path = Path("/home/users/nus/e1724738/_main/_private/NUS/_Proj1/SMOBench-CLEAN/Dataset")
    fusion_dirs = [
        "_myx_fusionWithGT",
        "_myx_fusionWoGT"
    ]
    
    print("Checking generated fusion files:")
    print("=" * 50)
    
    total_files = 0
    valid_files = 0
    
    for fusion_dir in fusion_dirs:
        fusion_path = base_path / fusion_dir
        if not fusion_path.exists():
            print(f"\nDirectory not found: {fusion_path}")
            continue
            
        print(f"\n{fusion_dir}:")
        
        for data_type_dir in fusion_path.iterdir():
            if data_type_dir.is_dir():
                print(f"\n  {data_type_dir.name}:")
                
                for fusion_file in data_type_dir.glob("*_Fusion_*.h5ad"):
                    total_files += 1
                    rel_path = fusion_file.relative_to(base_path)
                    
                    shape = check_file_shape(fusion_file)
                    if shape:
                        valid_files += 1
                        print(f"    {rel_path}: {shape[0]:,} cells, {shape[1]:,} features")
                    else:
                        print(f"    {rel_path}: cannot read")
    
    print(f"\nSummary:")
    print(f"Total files: {total_files}")
    print(f"Valid files: {valid_files}")
    print(f"Success rate: {valid_files/total_files*100:.1f}%" if total_files > 0 else "No files found")

if __name__ == "__main__":
    main()

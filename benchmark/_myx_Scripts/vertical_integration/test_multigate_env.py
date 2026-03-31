"""Quick sanity check for _Proj1_1_MultiGATE environment."""
import sys

def check(name, fn):
    try:
        result = fn()
        print(f"  [OK] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False

print("=== MultiGATE Environment Check ===\n")
ok = True

# 1. louvain
ok &= check("louvain import", lambda: __import__("louvain").__version__)

# 2. scanpy louvain clustering (the actual failure point)
def test_scanpy_louvain():
    import scanpy as sc
    import numpy as np
    adata = sc.AnnData(np.random.rand(50, 20))
    sc.pp.neighbors(adata)
    sc.tl.louvain(adata)
    return f"{len(adata.obs['louvain'].unique())} clusters on 50 cells"
ok &= check("scanpy.tl.louvain", test_scanpy_louvain)

# 3. TensorFlow + GPU
def test_tf_gpu():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    return f"TF {tf.__version__}, GPUs: {gpus}" if gpus else f"TF {tf.__version__}, NO GPU (will be slow!)"
ok &= check("TensorFlow GPU", test_tf_gpu)

# 4. MultiGATE import
ok &= check("MultiGATE import", lambda: (__import__("MultiGATE"), "OK")[1])

# 5. universal_clustering with louvain
def test_clustering():
    sys.path.insert(0, ".")
    from Utils.SMOBench_clustering import universal_clustering
    import scanpy as sc
    import numpy as np
    adata = sc.AnnData(np.random.rand(100, 30))
    adata.obsm["emb"] = np.random.rand(100, 10)
    sc.pp.neighbors(adata, use_rep="emb")
    adata = universal_clustering(adata, n_clusters=3, used_obsm="emb", method="louvain", key="louvain")
    return f"louvain: {len(adata.obs['louvain'].unique())} clusters"
ok &= check("universal_clustering(louvain)", test_clustering)

print(f"\n{'=== ALL PASSED ===' if ok else '=== SOME CHECKS FAILED ==='}")
sys.exit(0 if ok else 1)

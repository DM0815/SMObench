"""Generate clean spatial thumbnails for Fig1 overview - one representative slice per dataset."""
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUT = '_myx_Results/plots/fig1_thumbnails'
os.makedirs(OUT, exist_ok=True)

# Style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11

# PAL13 for cell types (matching project style)
PAL13 = ['#E64B35','#4DBBD5','#00A087','#3C5488','#F39B7F',
         '#8491B4','#91D1C2','#DC9172','#7E6148','#B09C85',
         '#E377C2','#BCBD22','#17BECF']

# Representative slice per dataset
DATASETS = {
    'Human_Lymph_Nodes': {
        'path': 'Dataset/withGT/RNA_ADT/Human_Lymph_Nodes/A1',
        'rna': 'adata_RNA.h5ad',
        'label': 'Spatial_Label',
        'title': 'Human Lymph Nodes',
        'modality': 'RNA+ADT',
        'species': 'Human',
        'gt': True,
        'sections': 2,
        'spots': '2.6-4.7k',
    },
    'Human_Tonsils': {
        'path': 'Dataset/withGT/RNA_ADT/Human_Tonsils/S1',
        'rna': 'adata_RNA.h5ad',
        'label': 'Spatial_Label',
        'title': 'Human Tonsils',
        'modality': 'RNA+ADT',
        'species': 'Human',
        'gt': True,
        'sections': 3,
        'spots': '3.2-4.4k',
    },
    'Mouse_Spleen': {
        'path': 'Dataset/woGT/RNA_ADT/Mouse_Spleen/Mouse_Spleen1',
        'rna': 'adata_RNA.h5ad',
        'label': None,
        'title': 'Mouse Spleen',
        'modality': 'RNA+ADT',
        'species': 'Mouse',
        'gt': False,
        'sections': 2,
        'spots': '2.8-3.1k',
    },
    'Mouse_Thymus': {
        'path': 'Dataset/woGT/RNA_ADT/Mouse_Thymus/Mouse_Thymus3',
        'rna': 'adata_RNA.h5ad',
        'label': None,
        'title': 'Mouse Thymus',
        'modality': 'RNA+ADT',
        'species': 'Mouse',
        'gt': False,
        'sections': 4,
        'spots': '2.9-4.0k',
    },
    'Mouse_Brain': {
        'path': 'Dataset/woGT/RNA_ATAC/Mouse_Brain/Mouse_Brain_H3K4me3',
        'rna': 'adata_RNA.h5ad',
        'label': None,
        'title': 'Mouse Brain',
        'modality': 'RNA+ATAC',
        'species': 'Mouse',
        'gt': False,
        'sections': 4,
        'spots': '8.7-14.8k',
    },
    'Mouse_Embryos_S1': {
        'path': 'Dataset/withGT/RNA_ATAC/Mouse_Embryos_S1/E11',
        'rna': 'adata_RNA.h5ad',
        'label': 'Spatial_Label',
        'title': 'Mouse Embryos S1',
        'modality': 'RNA+ATAC',
        'species': 'Mouse',
        'gt': True,
        'sections': 4,
        'spots': '5.2-17.8k',
    },
    'Mouse_Embryos_S2': {
        'path': 'Dataset/withGT/RNA_ATAC/Mouse_Embryos_S2/E11',
        'rna': 'adata_RNA.h5ad',
        'label': 'Spatial_Label',
        'title': 'Mouse Embryos S2',
        'modality': 'RNA+ATAC',
        'species': 'Mouse',
        'gt': True,
        'sections': 4,
        'spots': '4.9-15.6k',
    },
}

for ds_name, info in DATASETS.items():
    print(f"Processing {ds_name}...")
    h5ad_path = os.path.join(info['path'], info['rna'])
    if not os.path.exists(h5ad_path):
        print(f"  WARNING: {h5ad_path} not found, skipping")
        continue
    
    adata = sc.read_h5ad(h5ad_path)
    
    # Get spatial coordinates
    if 'spatial' in adata.obsm:
        coords = adata.obsm['spatial']
    elif 'X_spatial' in adata.obsm:
        coords = adata.obsm['X_spatial']
    else:
        print(f"  WARNING: no spatial coords for {ds_name}")
        continue
    
    x, y = coords[:, 0], coords[:, 1]
    
    # Determine colors
    if info['label'] and info['label'] in adata.obs.columns:
        # Use ground-truth labels
        cats = adata.obs[info['label']].astype('category')
        n_types = len(cats.cat.categories)
        if n_types <= len(PAL13):
            palette = PAL13[:n_types]
        else:
            cmap = plt.cm.get_cmap('tab20', n_types)
            palette = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(n_types)]
        color_map = {cat: palette[i] for i, cat in enumerate(cats.cat.categories)}
        colors = [color_map[c] for c in cats]
    elif 'RNA_clusters' in adata.obs.columns:
        # Mouse Brain has pre-computed clusters
        cats = adata.obs['RNA_clusters'].astype('category')
        n_types = len(cats.cat.categories)
        palette = PAL13[:n_types] if n_types <= len(PAL13) else \
            [matplotlib.colors.rgb2hex(plt.cm.get_cmap('tab20', n_types)(i)) for i in range(n_types)]
        color_map = {cat: palette[i] for i, cat in enumerate(cats.cat.categories)}
        colors = [color_map[c] for c in cats]
    else:
        # No GT and no pre-computed clusters: run leiden clustering
        print(f"  Running leiden clustering for {ds_name}...")
        adata_pp = adata.copy()
        sc.pp.normalize_total(adata_pp, target_sum=1e4)
        sc.pp.log1p(adata_pp)
        sc.pp.highly_variable_genes(adata_pp, n_top_genes=2000, flavor='seurat_v3')
        adata_pp = adata_pp[:, adata_pp.var.highly_variable].copy()
        sc.pp.scale(adata_pp, max_value=10)
        sc.tl.pca(adata_pp, n_comps=30)
        sc.pp.neighbors(adata_pp, n_neighbors=15, n_pcs=30)
        sc.tl.leiden(adata_pp, resolution=0.8)
        cats = adata_pp.obs['leiden'].astype('category')
        n_types = len(cats.cat.categories)
        palette = PAL13[:n_types] if n_types <= len(PAL13) else \
            [matplotlib.colors.rgb2hex(plt.cm.get_cmap('tab20', n_types)(i)) for i in range(n_types)]
        color_map = {cat: palette[i] for i, cat in enumerate(cats.cat.categories)}
        colors = [color_map[c] for c in cats]
    
    # Dynamic point size
    n_cells = len(adata)
    pt_size = max(3, min(25, 12000 / n_cells))
    
    # Plot - clean, no axes, no title, no legend
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(x, y, c=colors, s=pt_size, alpha=0.85, edgecolors='none', rasterized=True)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Tight crop
    x_margin = (x.max() - x.min()) * 0.05
    y_margin = (y.max() - y.min()) * 0.05
    ax.set_xlim(x.min() - x_margin, x.max() + x_margin)
    ax.set_ylim(y.min() - y_margin, y.max() + y_margin)
    
    # Invert y if needed (common for spatial data)
    if y.min() >= 0:
        ax.invert_yaxis()
    
    plt.tight_layout(pad=0.1)
    
    # Save PNG (high DPI for quality) and PDF
    out_png = os.path.join(OUT, f'{ds_name}.png')
    out_pdf = os.path.join(OUT, f'{ds_name}.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.05,
                facecolor='white', transparent=False)
    fig.savefig(out_pdf, bbox_inches='tight', pad_inches=0.05,
                facecolor='white')
    plt.close()
    print(f"  Saved: {out_png}")

print("\nDone! All thumbnails saved to", OUT)

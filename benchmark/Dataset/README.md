# Dataset Directory

Multi-modal spatial omics datasets for benchmarking spatial integration methods.

## Dataset Structure

### With Ground Truth (withGT/)
Datasets with cell type / region annotations (`Spatial_Label` column in `.obs`) for supervised evaluation (ARI, NMI, AMI, etc.).

**3M_Simulation**
- Triple-modality simulated data (RNA + ADT + ATAC), 1,296 spots, 5 spatial domains
- Source: SpatialGlue paper

**RNA_ADT (Protein)**
- **Human_Lymph_Nodes** (HLN): 2 samples (A1: 3,484 spots, D1: 3,359 spots). Manual H&E annotation.
  - Source: SpatialGlue paper (10x Visium CytAssist RNA + Protein)
- **Human_Tonsils** (HT): 3 samples (S1/S2/S3, ~4,300-4,500 spots). Has both `Spatial_Label` and `final_annot`.
  - Source: 10x Genomics Visium RNA + Protein

**RNA_ATAC (Chromatin)**
- **Mouse_Embryos_S1** (MISAR_S1): 4 timepoints (E11/E13/E15/E18), 1,263-2,129 spots
- **Mouse_Embryos_S2** (MISAR_S2): 4 timepoints (E11/E13/E15/E18), 1,353-2,248 spots
  - Source: MISAR-seq (RNA + ATAC co-profiling of mouse embryonic brain)

### Without Ground Truth (woGT/)
Datasets without reliable cell type annotations. Only unsupervised metrics (Moran's I, Jaccard similarity) can be used for evaluation.

**RNA_ADT (Protein)**
- **Mouse_Thymus**: 4 samples (Stereo-CITE-seq RNA + Protein). No annotations in data; cluster identity inferred from known thymus anatomy (medulla, cortex).
- **Mouse_Spleen**: 2 samples (SPOTS/10x Visium RNA + Protein). No annotations in data; cluster identity inferred post-hoc via marker genes/proteins.

**RNA_ATAC (Chromatin)**
- **Mouse_Brain**: 4 sub-datasets (ATAC, H3K27ac, H3K27me3, H3K4me3). Has algorithmically derived cluster labels (`RNA_clusters`, `*_clusters`) but NO manually annotated ground truth. Allen Brain Atlas used as reference only.
  - Source: spatial ATAC-RNA-seq / CUT&Tag-RNA-seq

## Ground Truth Evidence Summary

| Dataset | GT Available | GT Column | Evidence |
|---|---|---|---|
| 3M_Simulation | Yes | `Spatial_Label` (5 domains) | Synthetic data, GT known by design |
| Human_Lymph_Nodes | Yes | `Spatial_Label` (10-11 regions) | Manual H&E annotation (SpatialGlue paper Fig. 2e) |
| Human_Tonsils | Yes | `Spatial_Label` + `final_annot` (4-5 regions) | Annotated regions from 10x Genomics |
| Mouse_Embryos_S1/S2 | Yes | `Spatial_Label` (8-16 regions) | From MISAR-seq original dataset |
| Mouse_Thymus | No | - | SpatialGlue: only unsupervised metrics used |
| Mouse_Spleen | No | - | SpatialGlue: "macrophage subsets not available in original data annotations" |
| Mouse_Brain | No | algorithmic clusters only | SpatialGlue: "does not have an annotated ground truth", Allen Atlas as reference |

## Data Format

Each dataset contains:
- **adata_RNA.h5ad**: RNA expression matrix with spatial coordinates
- **adata_ADT.h5ad**: Protein expression matrix (for ADT datasets)
- **adata_ATAC.h5ad**: Chromatin accessibility matrix (for ATAC datasets)
- **adata_peaks_normalized.h5ad**: Normalized peaks (Mouse_Brain)
- **Spatial_Label**: Ground truth cell type annotations in `.obs` (withGT only)

## Download

Complete datasets available at:
https://drive.google.com/drive/u/1/folders/11zYh27BK9QuqU7zObApCYSzSEMqHS0G6

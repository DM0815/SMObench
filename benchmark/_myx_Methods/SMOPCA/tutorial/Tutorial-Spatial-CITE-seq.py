import logging
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
from SMOPCA.model import SMOPCA
from sklearn.cluster import KMeans 

for handler in logging.root.handlers[:]:  # avoid DEBUG level information in jupyter notebook
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO)  # use DEBUG for verbose information
warnings.filterwarnings('ignore')

data_file = h5py.File("/nfs/public/chenmo/PCA/SMOPCA-v2/data/RealData/SpatialCITEseq/Humantonsil_filtered.h5", 'r')
X1 = np.array(data_file['normalized_svg_count'])
X2 = np.array(data_file['normalized_protein_count'])
pos = np.array(data_file['pos'])
data_file.close()
print(X1.shape, X2.shape, pos.shape)

smopca = SMOPCA(Y_list=[X1.T, X2.T], Z_dim=20, pos=pos, intercept=False, omics_weight=False)
smopca.estimateParams(sigma_init_list=(1, 1), tol_sigma=2e-5, sigma_xtol_list=(1e-6, 1e-6), gamma_init=1, estimate_gamma=True)
z = smopca.calculatePosterior()
y_pred = KMeans(n_clusters=7, n_init=100).fit_predict(z)

adata_res = sc.AnnData(z)
adata_res.obsm['spatial'] = pos
adata_res.obs['pred'] = pd.Categorical(y_pred)
sc.pl.spatial(adata_res, color='pred', spot_size=1)

sc.pp.neighbors(adata_res, n_neighbors=100, use_rep='X', metric='euclidean')
sc.pl.umap(adata_res, color='pred')
sc.tl.umap(adata_res, min_dist=1)

library(rhdf5)
library(dplyr)
library(Seurat)
library(Matrix)
library(patchwork)
for (feat_type in c('gene', 'protein')) {
    h5_file <- H5Fopen(paste0("./data/Humantonsil_filtered.h5"))
    cell <- as.matrix(h5_file$cell)  # 2491
    if (feat_type == 'gene') {
        X <- as(h5_file$raw_gene_count, "sparseMatrix")
        gene <- as.matrix(h5_file$gene)
        rownames(X) <- gene
    } else {
        X <- as(h5_file$raw_protein_count, "sparseMatrix")
        protein <- as.matrix(h5_file$protein)
        rownames(X) <- protein
    }
    h5closeAll()
    colnames(X) <- cell
    y_pred <- as.matrix(read.csv("./data/smopca_pred_label.csv", row.names=1))
    seu <- CreateSeuratObject(counts = X, project = "diff_expr", min.cells = 0, min.features = 0)
    seu <- NormalizeData(seu, normalization.method = "LogNormalize", scale.factor = 10000)
    Idents(seu) <- y_pred
    y_unique <- c('0', '1', '2', '3', '4', '5', '6')
    res_folder <- "./res/" 
    for (g in y_unique) {
        de.markers <- FindMarkers(seu, ident.1 = g, ident.2 = NULL, only.pos = F)
        write.csv(de.markers, file=paste0(res_folder, paste0("cluster-", g, "-de-", feat_type, ".csv")))
    }
}


import numpy as np
import pandas as pd

top_feat_num = 10  # top gene num of each cluster
cluster_list = [0, 1, 2, 3, 4, 5, 6]
y_pred = pd.read_csv("./data/smopca_pred_label.csv", index_col=0).values.flatten()

for feat_type in ['gene', 'protein']:
    de_csv_list = []
    cluster_gene_dict = {}
    top_gene_list = []
    all_top_gene_list = []
    normed_df = pd.read_csv(f"./data/normed_{feat_type}_mat.csv", index_col=0)  # heatmap needs normalized raw data
    # print("checking raw data shape:", normed_df.shape, y_pred.shape)

    # find markers for each cluster
    for i in cluster_list:
        de_csv_i = pd.read_csv(f"./res/cluster-{i}-de-{feat_type}.csv", index_col=0)  # print this to see the sorted p values and genes
        # all_top_gene_list += list(de_csv_i.index[de_csv_i['p_val_adj'] < 0.05])
        # continue
        de_csv_list.append(de_csv_i)
        count = j = 0  # j for the current gene index
        while count < top_feat_num:
            if de_csv_i.index[j] in top_gene_list:
                # print(f"gene {de_csv_i.index[j]} already exist in {top_gene_list}, continue")
                j += 1
                continue
            if de_csv_i['p_val_adj'].iloc[j] > 0.05:
                # print(f"gene {de_csv_i.index[j]} p_val > 0.05, stop finding")
                break
            top_gene_list.append(de_csv_i.index[j])
            count += 1
            j += 1
        # print("cluster", i, de_csv_i.iloc[:j], end='\n\n')

    # find the subset expr matrix of interest
    data_list = []
    for top_gene in top_gene_list:
        expr_cluster_list = []
        expr_dat = np.array(normed_df.loc[top_gene])
        for i in cluster_list:
            expr_cluster = expr_dat[y_pred == i]
            expr_cluster_list.append(np.mean(expr_cluster))
        data_list.append(expr_cluster_list)
    data_mat = np.array(data_list)
    # print(data_mat.shape)
    data_df = pd.DataFrame(data=data_mat, index=top_gene_list, columns=[f'cluster_{i}' for i in cluster_list])
    # data_df = pd.DataFrame(data=data_mat.T, columns=top_gene_list, index=[f'cluster_{i}' for i in cluster_list])

    # change gene and protein names for plotting
    if feat_type == 'gene':
        new_ind = []
        for gene_name in data_df.index:
            if gene_name.count('.') == 0:
                new_ind.append(gene_name)
            else:
                new_gene_name = gene_name.replace('.', '-')
                new_ind.append(new_gene_name)
        data_df.index = new_ind
    elif feat_type == 'protein':
        new_ind = []
        for protein_name in data_df.index:
            if protein_name.count('human-mouse.Mac.2..Galectin.3') == 1:
                new_ind.append('Mac2')
            else:
                # new_ind.append(protein_name)
                new_ind.append(protein_name.split('.')[0])
        data_df.index = new_ind
    print(data_df)
    data_df.to_csv(f"./res/{feat_type}_dat_top={top_feat_num}.csv")


library(pheatmap)
myplot <- function(
	dat, ann, ann_color=NA, n.col=11, c_col=T, c_row=F, show_rownames=T, show_colnames=T, 
	breaks=seq(from=-max(abs(dat)), to=max(abs(dat)), len=n.col + 1), fontsize = 10
) {
	print(dat)
	pheatmap(
		dat, color = colorRampPalette(c("blue", "white", "red"))(n.col), clustering_distance_rows = "correlation",  clustering_distance_cols = "correlation",  clustering_method = "complete",
		cluster_cols = c_col, cluster_rows = c_row, fontsize = fontsize
	)
}

top_gene_num = 10
feat_type = 'gene'
dat_ = read.csv(paste0("./res/", feat_type, "_dat_top=", top_gene_num, ".csv"), row.names=1)
dat_ = t(scale(t(dat_), center = TRUE, scale = TRUE))
dat_[dat_ > 4] = 4
dat_[dat_ < -4] = -4

colnames(dat_)=c("0","1","2","3","4","5","6")

fontsize = 10
#pdf(paste0("./fig/", feat_type, "_fontsize=", fontsize, "_top=", top_gene_num, ".pdf"),width=11, height=3.7)
options(repr.plot.width=11,repr.plot.height=3.7)
myplot(t(dat_), ann=NA, ann_color=NA, c_col=F, c_row=F, show_colnames=F, show_rownames=T, fontsize=fontsize)#, breaks=break_ )
#dev.off()
tt=t(dat_)


feat_type = 'protein'
dat_ = read.csv(paste0("./res/", feat_type, "_dat_top=", top_gene_num, ".csv"), row.names=1)
dat_ = t(scale(t(dat_), center = TRUE, scale = TRUE))
dat_[dat_ > 4] = 4
dat_[dat_ < -4] = -4

colnames(dat_)=c("0","1","2","3","4","5","6")

fontsize = 10
#pdf(paste0("./fig/", feat_type, "_fontsize=", fontsize, "_top=", top_gene_num, ".pdf"),width=11, height=3.7)
options(repr.plot.width=11,repr.plot.height=3.7)
myplot(t(dat_), ann=NA, ann_color=NA, c_col=F, c_row=F, show_colnames=F, show_rownames=T, fontsize=fontsize)#, breaks=break_ )
#dev.off()
tt=t(dat_)


data_file = h5py.File("./data/Humantonsil_filtered.h5", 'r')
X = np.array(data_file['raw_protein_count'])
protein = np.array(data_file['protein']).astype(str)
pos = np.array(data_file['pos'])
data_file.close()
protein_rename = []
for name in protein:
    if name.count('Mac.2') == 1:
        protein_rename.append('Mac2')
    else:
        protein_rename.append(name.split('.')[0])
protein_rename = np.array(protein_rename)
plot_proteins = ['CD21', 'CD23', 'IgM', 'IgD', 'CD90', 'Mac2', 'CD3', 'CD4', 'CD45RA', 'CD32', 'CD9', 'CD171']
adata = sc.AnnData(X)
adata.obsm['spatial'] = pos
adata.var_names = protein_rename
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
fig, axes = plt.subplots(4, 3, figsize=(20, 25))
axes = axes.flatten()
for i, protein in enumerate(plot_proteins):
    sc.pl.spatial(adata, color=protein, ax=axes[i], show=False, title=protein, spot_size=1)
plt.tight_layout()
plt.show()
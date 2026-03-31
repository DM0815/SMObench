#!/usr/bin/env python3
"""
CM-GTC Synthetic Validation Experiment
=======================================
Validates CM-GTC metric behaviour on controlled synthetic data across 4 scenarios:

  1. Monotonic Embedding Destruction   - CM-GTC should decrease with increasing noise
  2. Modality-Selective Corruption     - CM-GTC should detect single-modality degradation
  3. ARI/NMI vs CM-GTC Discordance    - CM-GTC captures topology info missed by ARI/NMI
  4. Sample Size Robustness            - CM-GTC should stabilise for N >= 500

Data generation: sklearn.datasets.make_blobs (5 clusters, 2 modalities, 50 features each)

Usage:
    python cmgtc_synthetic_validation.py --root /path/to/SMOBench-CLEAN
    python cmgtc_synthetic_validation.py --root /path/to/SMOBench-CLEAN --seed 42
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Import CM-GTC
# ---------------------------------------------------------------------------

def setup_cmgtc_import(root_dir):
    """Import CMGTC class from storage."""
    candidates = [
        os.path.join(os.path.dirname(root_dir), 'storage', '_2_metric'),
        '/home/users/nus/e1724738/_main/_private/NUS/_Proj1/storage/_2_metric',
    ]
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, 'cm_gtc.py')):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            from cm_gtc import CMGTC
            return CMGTC
    raise FileNotFoundError("Cannot find cm_gtc.py")


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def generate_synthetic_data(n_samples=1000, n_clusters=5, n_features_per_mod=50,
                            random_state=42):
    """Generate synthetic multi-modal data with known cluster structure.

    Returns:
        latent: (N, 10) shared latent space
        mod1: (N, n_features) RNA-like modality
        mod2: (N, n_features) ADT-like modality
        labels: (N,) ground truth cluster labels
    """
    rng = np.random.RandomState(random_state)

    # Shared latent structure
    latent, labels = make_blobs(
        n_samples=n_samples, n_features=10, centers=n_clusters,
        cluster_std=1.0, random_state=random_state
    )

    # Generate modality data via random linear projections from latent
    W1 = rng.randn(10, n_features_per_mod) * 0.5
    W2 = rng.randn(10, n_features_per_mod) * 0.5

    mod1 = latent @ W1 + rng.randn(n_samples, n_features_per_mod) * 0.3
    mod2 = latent @ W2 + rng.randn(n_samples, n_features_per_mod) * 0.3

    # Make non-negative (RNA-like / ADT-like)
    mod1 = mod1 - mod1.min() + 0.1
    mod2 = mod2 - mod2.min() + 0.1

    return latent, mod1, mod2, labels


def compute_ari_nmi(labels_true, embedding, n_clusters=5):
    """Compute ARI and NMI via KMeans clustering on the embedding."""
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels_pred = km.fit_predict(embedding)
    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    return ari, nmi


# ---------------------------------------------------------------------------
# Scenario 1: Monotonic Embedding Destruction
# ---------------------------------------------------------------------------

def scenario1_monotonic_destruction(CMGTC_cls, n_samples=1000, n_clusters=5,
                                    seed=42, n_repeats=5):
    """Add increasing Gaussian noise to a perfect joint embedding.

    Expected: CM-GTC monotonically decreases; Spearman(sigma, CM-GTC) < -0.9
    """
    print("\n" + "=" * 70)
    print("SCENARIO 1: Monotonic Embedding Destruction")
    print("=" * 70)

    sigma_values = np.round(np.arange(0, 2.05, 0.1), 1)
    rows = []

    for rep in range(n_repeats):
        rs = seed + rep
        latent, mod1, mod2, labels = generate_synthetic_data(
            n_samples=n_samples, n_clusters=n_clusters, random_state=rs
        )
        evaluator = CMGTC_cls(
            similarity_metric='cosine', correlation_metric='spearman',
            aggregation_strategy='min', verbose=False
        )

        for sigma in sigma_values:
            rng = np.random.RandomState(rs * 100 + int(sigma * 10))
            noise = rng.randn(*latent.shape) * sigma
            noisy_embedding = latent + noise

            modalities = {'rna': mod1, 'adt': mod2}
            score, _ = evaluator.compute_cm_gtc(
                noisy_embedding, modalities, auto_preprocess=False
            )

            ari, nmi = compute_ari_nmi(labels, noisy_embedding, n_clusters)

            rows.append({
                'Scenario': 'S1_Monotonic',
                'Repeat': rep,
                'Sigma': sigma,
                'CM_GTC': score,
                'ARI': ari,
                'NMI': nmi,
                'N_Samples': n_samples,
            })
            print(f"  rep={rep} sigma={sigma:.1f} CM-GTC={score:.4f} "
                  f"ARI={ari:.4f} NMI={nmi:.4f}")

    df = pd.DataFrame(rows)

    # Validate: Spearman correlation between sigma and mean CM-GTC
    mean_per_sigma = df.groupby('Sigma')['CM_GTC'].mean()
    spearman_corr, spearman_p = stats.spearmanr(
        mean_per_sigma.index, mean_per_sigma.values
    )
    print(f"\n  Spearman(sigma, CM-GTC) = {spearman_corr:.4f} (p={spearman_p:.2e})")
    print(f"  PASS: {spearman_corr < -0.9}")

    return df


# ---------------------------------------------------------------------------
# Scenario 2: Modality-Selective Corruption
# ---------------------------------------------------------------------------

def scenario2_modality_selective(CMGTC_cls, n_samples=1000, n_clusters=5,
                                 seed=42, n_repeats=5):
    """Corrupt only one modality; keep the other intact.

    Expected: CM-GTC drops when either modality is corrupted;
              ARI/NMI may stay high if the embedding is not re-computed.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 2: Modality-Selective Corruption")
    print("=" * 70)

    sigma_values = np.round(np.arange(0, 2.05, 0.2), 1)
    rows = []

    for rep in range(n_repeats):
        rs = seed + rep
        latent, mod1, mod2, labels = generate_synthetic_data(
            n_samples=n_samples, n_clusters=n_clusters, random_state=rs
        )
        evaluator = CMGTC_cls(
            similarity_metric='cosine', correlation_metric='spearman',
            aggregation_strategy='min', verbose=False
        )

        # Use the perfect latent as joint embedding (simulate ideal integration)
        joint_embedding = latent.copy()

        for sigma in sigma_values:
            rng = np.random.RandomState(rs * 100 + int(sigma * 10))

            for corrupt_mod in ['rna', 'adt', 'none']:
                if corrupt_mod == 'rna':
                    m1 = mod1 + rng.randn(*mod1.shape) * sigma * mod1.std()
                    m2 = mod2.copy()
                elif corrupt_mod == 'adt':
                    m1 = mod1.copy()
                    m2 = mod2 + rng.randn(*mod2.shape) * sigma * mod2.std()
                else:
                    m1 = mod1.copy()
                    m2 = mod2.copy()

                modalities = {'rna': m1, 'adt': m2}
                score, details = evaluator.compute_cm_gtc(
                    joint_embedding, modalities, auto_preprocess=False
                )

                # ARI/NMI on the joint embedding (unchanged)
                ari, nmi = compute_ari_nmi(labels, joint_embedding, n_clusters)

                # Per-modality scores
                per_mod = details.get('per_modality_stats', {})
                rna_mean = per_mod.get('rna', {}).get('mean_consistency', np.nan)
                adt_mean = per_mod.get('adt', {}).get('mean_consistency', np.nan)

                rows.append({
                    'Scenario': 'S2_Selective',
                    'Repeat': rep,
                    'Sigma': sigma,
                    'Corrupted': corrupt_mod,
                    'CM_GTC': score,
                    'CM_GTC_rna': rna_mean,
                    'CM_GTC_adt': adt_mean,
                    'ARI': ari,
                    'NMI': nmi,
                    'N_Samples': n_samples,
                })

            if sigma in [0, 0.4, 1.0, 2.0]:
                sub = [r for r in rows if r['Repeat'] == rep and r['Sigma'] == sigma]
                for r in sub:
                    print(f"  rep={rep} sigma={sigma:.1f} "
                          f"corrupt={r['Corrupted']:4s} CM-GTC={r['CM_GTC']:.4f} "
                          f"ARI={r['ARI']:.4f}")

    df = pd.DataFrame(rows)

    # Validate: at high sigma, corrupted modality CM-GTC < baseline
    baseline = df[(df['Corrupted'] == 'none') & (df['Sigma'] == 0)]['CM_GTC'].mean()
    corrupted_high = df[(df['Corrupted'] != 'none') & (df['Sigma'] >= 1.0)]['CM_GTC'].mean()
    ari_baseline = df[(df['Corrupted'] == 'none') & (df['Sigma'] == 0)]['ARI'].mean()
    ari_corrupted = df[(df['Corrupted'] != 'none') & (df['Sigma'] >= 1.0)]['ARI'].mean()

    print(f"\n  CM-GTC baseline={baseline:.4f}, corrupted@high_sigma={corrupted_high:.4f}")
    print(f"  ARI   baseline={ari_baseline:.4f}, corrupted@high_sigma={ari_corrupted:.4f}")
    print(f"  CM-GTC detects corruption: {corrupted_high < baseline * 0.8}")
    print(f"  ARI unchanged (joint unchanged): {abs(ari_corrupted - ari_baseline) < 0.05}")

    return df


# ---------------------------------------------------------------------------
# Scenario 3: ARI/NMI vs CM-GTC Discordance
# ---------------------------------------------------------------------------

def scenario3_discordance(CMGTC_cls, n_samples=1000, n_clusters=5,
                           seed=42, n_repeats=10):
    """Construct embeddings where clustering metrics and topology metrics disagree.

    (A) Cluster-correct but topology-shuffled: high ARI/NMI, low CM-GTC
        - Snap each point to its cluster center + tiny noise → destroys
          intra-cluster neighbor ordering but KMeans still correct.
    (B) Cluster-slightly-off but topology-preserved: lower ARI/NMI, higher CM-GTC
        - Add moderate noise so boundary points cross cluster borders →
          KMeans makes mistakes, but neighbor ordering stays close to truth.

    Uses overlapping clusters (3D latent, cluster_std=2.5) so that KMeans
    is not trivially perfect.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 3: ARI/NMI vs CM-GTC Discordance")
    print("=" * 70)

    rows = []

    for rep in range(n_repeats):
        rs = seed + rep
        rng = np.random.RandomState(rs)

        # Moderately separated clusters
        latent_3d, labels = make_blobs(
            n_samples=n_samples, n_features=3, centers=n_clusters,
            cluster_std=1.5, random_state=rs
        )

        # Project to 10D for consistency with modalities
        W_proj = rng.randn(3, 10) * 0.5
        latent = latent_3d @ W_proj

        # Generate modality data from the 10D latent
        W1 = rng.randn(10, 50) * 0.5
        W2 = rng.randn(10, 50) * 0.5
        mod1 = latent @ W1 + rng.randn(n_samples, 50) * 0.3
        mod2 = latent @ W2 + rng.randn(n_samples, 50) * 0.3
        mod1 = mod1 - mod1.min() + 0.1
        mod2 = mod2 - mod2.min() + 0.1

        evaluator = CMGTC_cls(
            similarity_metric='cosine', correlation_metric='spearman',
            aggregation_strategy='min', verbose=False
        )
        modalities = {'rna': mod1, 'adt': mod2}

        # --- Type A: Cluster-correct, topology-DESTROYED ---
        # Generate completely new random embedding per cluster (no relation to
        # original latent), but place clusters far apart so KMeans still works.
        # Global topology (which point is near which) is completely random.
        centers_new = rng.randn(n_clusters, latent.shape[1]) * 10  # well-separated
        embedding_A = np.zeros_like(latent)
        for c in range(n_clusters):
            mask = labels == c
            n_c = mask.sum()
            # Random points in a tight ball around new center
            embedding_A[mask] = centers_new[c] + rng.randn(n_c, latent.shape[1]) * 0.5

        # --- Type B: Cluster-blurred, topology-PRESERVED ---
        # Keep original structure (preserves topology) but add heavy noise
        # so cluster boundaries blur significantly → ARI drops to ~0.4-0.6
        embedding_B = latent + rng.randn(*latent.shape) * 5.0

        # Compute metrics
        for emb_type, embedding in [('A_cluster_good', embedding_A),
                                     ('B_topology_good', embedding_B)]:
            score, _ = evaluator.compute_cm_gtc(
                embedding, modalities, auto_preprocess=False
            )
            ari, nmi = compute_ari_nmi(labels, embedding, n_clusters)

            rows.append({
                'Scenario': 'S3_Discordance',
                'Repeat': rep,
                'Type': emb_type,
                'CM_GTC': score,
                'ARI': ari,
                'NMI': nmi,
                'N_Samples': n_samples,
            })

        if rep < 3:
            sub = [r for r in rows if r['Repeat'] == rep]
            for r in sub[-2:]:
                print(f"  rep={rep} {r['Type']:20s} "
                      f"CM-GTC={r['CM_GTC']:.4f} ARI={r['ARI']:.4f} NMI={r['NMI']:.4f}")

    df = pd.DataFrame(rows)

    # Validate: A should have higher ARI but lower CM-GTC than B
    A = df[df['Type'] == 'A_cluster_good']
    B = df[df['Type'] == 'B_topology_good']

    print(f"\n  Type A (cluster-good):  ARI={A['ARI'].mean():.4f}  "
          f"NMI={A['NMI'].mean():.4f}  CM-GTC={A['CM_GTC'].mean():.4f}")
    print(f"  Type B (topology-good): ARI={B['ARI'].mean():.4f}  "
          f"NMI={B['NMI'].mean():.4f}  CM-GTC={B['CM_GTC'].mean():.4f}")
    print(f"  Discordance (A.ARI > B.ARI): {A['ARI'].mean() > B['ARI'].mean()}")
    print(f"  Discordance (A.CM-GTC < B.CM-GTC): {A['CM_GTC'].mean() < B['CM_GTC'].mean()}")

    return df


# ---------------------------------------------------------------------------
# Scenario 4: Sample Size Robustness
# ---------------------------------------------------------------------------

def scenario4_sample_size(CMGTC_cls, n_clusters=5, seed=42, n_repeats=10):
    """Vary N and measure CM-GTC stability.

    Generate ONE large pool (N=6000) with a fixed generative model, then
    draw multiple independent subsamples at each size.  CV is computed
    across subsamples from the SAME underlying data, isolating the pure
    effect of sample size on CM-GTC stability.

    Expected: CV < 0.1 for N >= 500
    """
    print("\n" + "=" * 70)
    print("SCENARIO 4: Sample Size Robustness")
    print("=" * 70)

    sample_sizes = [100, 500, 1000, 2000, 5000]
    N_pool = 6000
    rows = []

    # Generate ONE fixed pool
    latent_pool, mod1_pool, mod2_pool, labels_pool = generate_synthetic_data(
        n_samples=N_pool, n_clusters=n_clusters, random_state=seed
    )

    evaluator = CMGTC_cls(
        similarity_metric='cosine', correlation_metric='spearman',
        aggregation_strategy='min', verbose=False
    )

    for N in sample_sizes:
        for rep in range(n_repeats):
            # Each rep draws a DIFFERENT subsample from the SAME pool
            rng = np.random.RandomState(seed * 1000 + N * 100 + rep)
            idx = rng.choice(N_pool, size=N, replace=False)
            idx.sort()

            latent_sub = latent_pool[idx]
            mod1_sub = mod1_pool[idx]
            mod2_sub = mod2_pool[idx]

            modalities = {'rna': mod1_sub, 'adt': mod2_sub}
            score, _ = evaluator.compute_cm_gtc(
                latent_sub, modalities, auto_preprocess=False
            )

            rows.append({
                'Scenario': 'S4_SampleSize',
                'Repeat': rep,
                'N_Samples': N,
                'CM_GTC': score,
            })

    df = pd.DataFrame(rows)

    for N in sample_sizes:
        sub = df[df['N_Samples'] == N]
        mean_s = sub['CM_GTC'].mean()
        std_s = sub['CM_GTC'].std()
        cv = std_s / mean_s if mean_s > 0 else np.inf
        print(f"  N={N:5d}: CM-GTC={mean_s:.4f} +/- {std_s:.4f} (CV={cv:.4f})")

    # Validate: CV < 0.1 for N >= 500
    for N in [500, 1000, 2000, 5000]:
        sub = df[df['N_Samples'] == N]
        cv = sub['CM_GTC'].std() / sub['CM_GTC'].mean() if sub['CM_GTC'].mean() > 0 else np.inf
        print(f"  N={N}: CV={cv:.4f} {'PASS' if cv < 0.1 else 'FAIL'}")

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def create_validation_figure(df1, df2, df3, df4, output_path):
    """Create 2x2 panel figure summarising all 4 scenarios."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('CM-GTC Synthetic Validation', fontsize=14, fontweight='bold')

    # --- Panel A: Monotonic Destruction ---
    ax = axes[0, 0]
    mean_df = df1.groupby('Sigma').agg({'CM_GTC': ['mean', 'std'],
                                         'ARI': ['mean', 'std'],
                                         'NMI': ['mean', 'std']}).reset_index()
    sigmas = mean_df['Sigma'].values
    cmgtc_mean = mean_df[('CM_GTC', 'mean')].values
    cmgtc_std = mean_df[('CM_GTC', 'std')].values
    ari_mean = mean_df[('ARI', 'mean')].values
    nmi_mean = mean_df[('NMI', 'mean')].values

    ax.plot(sigmas, cmgtc_mean, 'o-', color='#2196F3', label='CM-GTC', linewidth=2)
    ax.fill_between(sigmas, cmgtc_mean - cmgtc_std, cmgtc_mean + cmgtc_std,
                    alpha=0.2, color='#2196F3')
    ax.plot(sigmas, ari_mean, 's--', color='#FF9800', label='ARI', linewidth=1.5)
    ax.plot(sigmas, nmi_mean, '^--', color='#4CAF50', label='NMI', linewidth=1.5)

    # Compute and annotate Spearman
    rho, _ = stats.spearmanr(sigmas, cmgtc_mean)
    ax.set_title(f'(a) Monotonic Destruction\n'
                 r'$\rho$(noise, CM-GTC) = ' + f'{rho:.3f}', fontsize=11)
    ax.set_xlabel('Noise level ($\\sigma$)')
    ax.set_ylabel('Score')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel B: Modality-Selective Corruption ---
    ax = axes[0, 1]
    for corrupt_type, color, marker in [('none', '#4CAF50', 'o'),
                                         ('rna', '#F44336', 's'),
                                         ('adt', '#FF9800', '^')]:
        sub = df2[df2['Corrupted'] == corrupt_type]
        mean_sub = sub.groupby('Sigma')['CM_GTC'].agg(['mean', 'std']).reset_index()
        ax.plot(mean_sub['Sigma'], mean_sub['mean'], f'{marker}-',
                color=color, label=f'Corrupt: {corrupt_type}', linewidth=2)
        ax.fill_between(mean_sub['Sigma'],
                        mean_sub['mean'] - mean_sub['std'],
                        mean_sub['mean'] + mean_sub['std'],
                        alpha=0.15, color=color)

    ax.set_title('(b) Modality-Selective Corruption', fontsize=11)
    ax.set_xlabel('Noise level ($\\sigma$)')
    ax.set_ylabel('CM-GTC')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel C: Discordance ---
    ax = axes[1, 0]
    A = df3[df3['Type'] == 'A_cluster_good']
    B = df3[df3['Type'] == 'B_topology_good']

    x_pos = [0, 1]
    width = 0.35

    # ARI bars
    ax.bar([p - width/2 for p in x_pos],
           [A['ARI'].mean(), B['ARI'].mean()],
           width, color='#FF9800', alpha=0.8, label='ARI',
           yerr=[A['ARI'].std(), B['ARI'].std()], capsize=4)
    # CM-GTC bars
    ax.bar([p + width/2 for p in x_pos],
           [A['CM_GTC'].mean(), B['CM_GTC'].mean()],
           width, color='#2196F3', alpha=0.8, label='CM-GTC',
           yerr=[A['CM_GTC'].std(), B['CM_GTC'].std()], capsize=4)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(['(A) Cluster-correct\nTopology-shuffled',
                        '(B) Topology-preserved\nCluster-slightly-off'],
                       fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('(c) ARI vs CM-GTC Discordance', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel D: Sample Size Robustness ---
    ax = axes[1, 1]
    size_summary = df4.groupby('N_Samples')['CM_GTC'].agg(['mean', 'std']).reset_index()
    sizes = size_summary['N_Samples'].values
    means = size_summary['mean'].values
    stds = size_summary['std'].values
    cvs = stds / means

    ax.errorbar(sizes, means, yerr=stds, fmt='o-', color='#2196F3',
                linewidth=2, capsize=5, markersize=8)

    ax2 = ax.twinx()
    ax2.bar(sizes, cvs, width=np.array(sizes) * 0.15, alpha=0.3,
            color='#F44336', label='CV')
    ax2.axhline(y=0.1, color='#F44336', linestyle='--', alpha=0.5, label='CV=0.1')
    ax2.set_ylabel('CV (Coefficient of Variation)', color='#F44336')

    ax.set_xlabel('Sample size (N)')
    ax.set_ylabel('CM-GTC', color='#2196F3')
    ax.set_title('(d) Sample Size Robustness', fontsize=11)
    ax.set_xscale('log')
    ax2.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='CM-GTC Synthetic Validation Experiment'
    )
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory of SMOBench-CLEAN')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--n_repeats', type=int, default=5,
                        help='Number of repeats per condition')
    parser.add_argument('--skip_plot', action='store_true',
                        help='Skip figure generation')
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(args.root)

    CMGTC_cls = setup_cmgtc_import(root)

    output_dir = os.path.join(root, '_myx_Results', 'evaluation', 'cmgtc_validation')
    fig_dir = os.path.join(root, '_myx_Results', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Root:       {root}")
    print(f"Output:     {output_dir}")
    print(f"Seed:       {args.seed}")
    print(f"N_Samples:  {args.n_samples}")
    print(f"N_Repeats:  {args.n_repeats}")

    total_start = time.time()

    # --- Scenario 1 ---
    df1 = scenario1_monotonic_destruction(
        CMGTC_cls, n_samples=args.n_samples, seed=args.seed,
        n_repeats=args.n_repeats
    )
    df1.to_csv(os.path.join(output_dir, 'scenario1_monotonic.csv'), index=False)
    print(f"  Saved scenario1_monotonic.csv ({len(df1)} rows)")

    # --- Scenario 2 ---
    df2 = scenario2_modality_selective(
        CMGTC_cls, n_samples=args.n_samples, seed=args.seed,
        n_repeats=args.n_repeats
    )
    df2.to_csv(os.path.join(output_dir, 'scenario2_selective.csv'), index=False)
    print(f"  Saved scenario2_selective.csv ({len(df2)} rows)")

    # --- Scenario 3 ---
    df3 = scenario3_discordance(
        CMGTC_cls, n_samples=args.n_samples, seed=args.seed,
        n_repeats=max(args.n_repeats, 10)
    )
    df3.to_csv(os.path.join(output_dir, 'scenario3_discordance.csv'), index=False)
    print(f"  Saved scenario3_discordance.csv ({len(df3)} rows)")

    # --- Scenario 4 ---
    df4 = scenario4_sample_size(
        CMGTC_cls, seed=args.seed,
        n_repeats=max(args.n_repeats, 10)
    )
    df4.to_csv(os.path.join(output_dir, 'scenario4_samplesize.csv'), index=False)
    print(f"  Saved scenario4_samplesize.csv ({len(df4)} rows)")

    # --- Summary CSV ---
    summary_rows = []

    # S1 summary
    mean_per_sigma = df1.groupby('Sigma')['CM_GTC'].mean()
    rho1, p1 = stats.spearmanr(mean_per_sigma.index, mean_per_sigma.values)
    summary_rows.append({
        'Scenario': 'S1_Monotonic', 'Metric': 'Spearman(sigma,CM-GTC)',
        'Value': rho1, 'Threshold': -0.9,
        'Pass': rho1 < -0.9
    })

    # S2 summary
    baseline_cmgtc = df2[(df2['Corrupted'] == 'none') & (df2['Sigma'] == 0)]['CM_GTC'].mean()
    corrupt_cmgtc = df2[(df2['Corrupted'] != 'none') & (df2['Sigma'] >= 1.0)]['CM_GTC'].mean()
    summary_rows.append({
        'Scenario': 'S2_Selective', 'Metric': 'CM-GTC_drop_ratio',
        'Value': corrupt_cmgtc / baseline_cmgtc if baseline_cmgtc > 0 else np.nan,
        'Threshold': 0.8,
        'Pass': corrupt_cmgtc < baseline_cmgtc * 0.8
    })

    # S3 summary
    A_ari = df3[df3['Type'] == 'A_cluster_good']['ARI'].mean()
    B_ari = df3[df3['Type'] == 'B_topology_good']['ARI'].mean()
    A_cmgtc = df3[df3['Type'] == 'A_cluster_good']['CM_GTC'].mean()
    B_cmgtc = df3[df3['Type'] == 'B_topology_good']['CM_GTC'].mean()
    summary_rows.append({
        'Scenario': 'S3_Discordance', 'Metric': 'A.ARI>B.ARI AND A.CMGTC<B.CMGTC',
        'Value': f'ARI:{A_ari:.3f}>{B_ari:.3f}, CMGTC:{A_cmgtc:.3f}<{B_cmgtc:.3f}',
        'Threshold': 'both true',
        'Pass': (A_ari > B_ari) and (A_cmgtc < B_cmgtc)
    })

    # S4 summary
    for N in [500, 1000, 2000, 5000]:
        sub = df4[df4['N_Samples'] == N]
        cv = sub['CM_GTC'].std() / sub['CM_GTC'].mean() if sub['CM_GTC'].mean() > 0 else np.inf
        summary_rows.append({
            'Scenario': f'S4_N={N}', 'Metric': 'CV',
            'Value': cv, 'Threshold': 0.1,
            'Pass': cv < 0.1
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, 'validation_summary.csv'), index=False)
    print(f"\n{'=' * 70}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 70}")
    print(summary_df.to_string(index=False))

    n_pass = summary_df['Pass'].sum()
    n_total = len(summary_df)
    print(f"\nOverall: {n_pass}/{n_total} checks passed")

    # --- Figure ---
    if not args.skip_plot:
        fig_path = os.path.join(fig_dir, 'cmgtc_validation.pdf')
        try:
            create_validation_figure(df1, df2, df3, df4, fig_path)
        except Exception as e:
            print(f"Figure generation failed: {e}")

    elapsed = time.time() - total_start
    print(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == '__main__':
    main()

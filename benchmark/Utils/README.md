# Utilities

Shared utility functions and interfaces for SMOBench framework.

## Core Utilities

### SMOBench_clustering.py
Universal clustering interface supporting multiple algorithms:

**Clustering Methods:**
- **Leiden**: Community detection algorithm
- **Louvain**: Modularity-based clustering  
- **K-means**: Centroid-based clustering
- **Mclust**: Model-based clustering (R interface)

**Features:**
- Automatic resolution tuning for graph-based methods
- Standardized input/output format
- Integration with AnnData objects
- Consistent clustering across all methods
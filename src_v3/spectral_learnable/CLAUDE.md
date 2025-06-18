# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Running the Model
```bash
# Basic run with BCE loss (default for most datasets)
python main.py --dataset ml-100k --filter ui

# ML-100k with optimized settings
python main.py --dataset ml-100k --filter uib --u_n_eigen 30 --i_n_eigen 50 --b_n_eigen 200

# Yelp2018 with user-bipartite filter
python main.py --dataset yelp2018 --filter ub --u_n_eigen 250 --i_n_eigen 400 --b_n_eigen 600 --patience 10 --epochs 50 --lr 0.1

# Using multi-hop propagation (2-hop by default for all filters)
python main.py --dataset ml-100k --filter uib --n_hops 2 --hop_weight 0.7

# 3-hop propagation for sparse datasets (all filters)
python main.py --dataset gowalla --filter uib --n_hops 3

# Test user-only with multi-hop to see pure user-based propagation
python main.py --dataset lastfm --filter u --n_hops 2 --hop_weight 0.5

# Test item-only with multi-hop for item similarity propagation
python main.py --dataset ml-100k --filter i --n_hops 3

# Using presets for quick configuration
python main.py --dataset ml-100k --preset fast      # Quick testing
python main.py --dataset ml-100k --preset balanced  # Standard run
python main.py --dataset ml-100k --preset quality   # High quality results
python main.py --dataset ml-100k --preset experimental  # Ensemble filters
```

### Hyperparameter Search
```bash
# Search for optimal user filter design and initialization
python hp_search.py --view u --dataset ml-100k --epochs 30

# Search for item filter with specific designs
python hp_search.py --view i --dataset gowalla --filter_designs chebyshev jacobi legendre

# Search for bipartite filter with custom settings
python hp_search.py --view b --dataset yelp2018 --base_filter uib --n_eigen 400
```

### Filter Options
- `--filter`: u (user), i (item), b (bipartite), ui, ub, or uib
- `--user_filter_design`: Filter type for user similarity (default: multiscale)
- `--item_filter_design`: Filter type for item similarity (default: chebyshev)
- `--bipartite_filter_design`: Filter type for bipartite graph (default: original)

Available filter designs: original, spectral_basis, enhanced_basis, chebyshev, jacobi, legendre, laguerre, hermite, bernstein, universal_polynomial, bandstop, adaptive_bandstop, parametric, multiscale, harmonic, golden, ensemble

## Code Architecture

### Core Components

1. **SpectralCF Model** (`model.py`)
   - Implements spectral collaborative filtering with multiple filter types
   - Key classes: `UniversalSpectralCF`, `UserSpecificSpectralCF`
   - Computes eigendecompositions and applies spectral filters
   - Supports user, item, and bipartite graph similarities

2. **Filter System** (`filters.py`)
   - 16 different spectral filter implementations
   - Each filter applies polynomial transformations to eigenvalues
   - Filters can be combined (u/i/b/ui/ub/uib) for better performance

3. **Training Procedure** (`procedure.py`)
   - BCE (Binary Cross-Entropy) loss with negative sampling
   - Evaluation with recall@k and ndcg@k metrics
   - Early stopping based on validation performance

4. **Data Loading** (`dataloader.py`)
   - Supports datasets: ml-100k, lastfm, gowalla, yelp2018, amazon-book
   - Creates train/validation/test splits
   - Handles sparse user-item interaction matrices

### Key Design Features

1. **Eigenvalue Configuration**
   - Set to 0 for auto-calculation based on dataset size
   - Manual specification for fine-tuning: `--u_n_eigen`, `--i_n_eigen`, `--b_n_eigen`

2. **Personalization Dimensions**
   - User/Item/Bipartite specific dimensions for learned components
   - Controlled via `--user_personalization_dim`, etc.

3. **Filter Initialization**
   - Different patterns: smooth, sharp, bandpass, golden_036, butterworth, gaussian, band_stop, notch
   - Set via `--user_init_filter`, `--item_init_filter`, `--bipartite_init_filter`

4. **Multi-hop Propagation** (NEW)
   - All filter types (user, item, bipartite) now support 1, 2, or 3-hop propagation
   - `--n_hops`: Controls propagation depth for all filters (default: 2)
   - `--hop_weight`: Balances multi-hop vs direct signals for 2-hop (default: 0.7)
   
   **User-based multi-hop**: user → similar users → users similar to those → items
   **Item-based multi-hop**: user's items → similar items → items similar to those
   **Bipartite multi-hop**: user → item → user → item (through bipartite graph)
   
   - 2-hop: Captures collaborative patterns through indirect connections
   - 3-hop: Extended propagation for very sparse datasets
   - Often improves NDCG (ranking quality) even when recall is similar

### Dataset Locations
All datasets are stored in `../data/` relative to src_v3:
- `../data/ml-100k/`: MovieLens 100K
- `../data/gowalla/`: Gowalla check-ins
- `../data/yelp2018/`: Yelp reviews
- `../data/amazon-book/`: Amazon book reviews
- `../data/lastfm/`: Last.fm music data

### Performance Notes
- Eigendecomposition is computationally expensive but computed only once
- Results are not cached between runs (unlike v6)
- BCE loss typically works better than MSE for this version
- Filter combination (uib) often gives best results but is slower
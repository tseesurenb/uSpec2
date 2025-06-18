# SpectralCF Learnable v1

Clean implementation of learnable spectral collaborative filtering with per-view optimization.

## Key Features

1. **Learnable Spectral Filters**
   - Bernstein polynomials (universal approximator)
   - Chebyshev polynomials (optimal approximation)
   - Spectral basis (frequency-selective)

2. **Per-View Hyperparameters**
   - Separate learning rates for user/item/bipartite filters
   - Separate weight decay for regularization
   - Adaptive to different eigenvalue scales

3. **Flexible Architecture**
   - Choose which views to use (u, i, b, ui, ub, ib, uib)
   - Configurable filter order/complexity
   - Multiple initialization strategies

## Quick Start

```bash
# Basic training
python main.py --dataset ml-100k --filter uib

# With custom hyperparameters
python main.py --dataset ml-100k \
    --filter uib \
    --filter_type bernstein \
    --user_lr 0.1 --item_lr 0.01 --bipartite_lr 0.05 \
    --epochs 100

# Try different filter types
python main.py --dataset ml-100k --filter_type chebyshev
python main.py --dataset ml-100k --filter_type spectral_basis

# Log filter evolution
python main.py --dataset ml-100k --log_filters --exp_name filter_evolution
```

## Hyperparameter Guidelines

### For Small Datasets (ML-100K)
- User LR: 0.1 (fewer users, more exploration)
- Item LR: 0.01 (more items, need stability)
- Bipartite LR: 0.05 (balanced)

### For Large Datasets (Gowalla, Yelp)
- User LR: 0.01 (many users, careful updates)
- Item LR: 0.001 (many items, very careful)
- Bipartite LR: 0.005 (stability first)

## Filter Types

1. **Bernstein** (Recommended)
   - Smooth, bounded output
   - Good for general use
   - Stable training

2. **Chebyshev**
   - Fast convergence
   - Good for sharp cutoffs
   - May need careful initialization

3. **Spectral Basis**
   - Best for frequency-selective filtering
   - More parameters
   - Good when you know the spectral pattern

## Experiment Tracking

Results are saved to `experiments/` with:
- Filter response plots (if --log_filters)
- Best model checkpoint (if --save_model)
- Training logs

## Advanced Usage

```bash
# Use cosine annealing scheduler
python main.py --dataset ml-100k --scheduler cosine

# Try AdamW optimizer with higher decay
python main.py --dataset ml-100k --optimizer adamw \
    --user_decay 1e-3 --item_decay 5e-3

# Different initializations
python main.py --dataset ml-100k \
    --user_init smooth --item_init sharp --bipartite_init lowpass
```
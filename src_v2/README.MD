# Usage Examples for All Model Types

## Quick Start Examples

### Simple Model (Minimal, Fast)
```bash
# Default configuration - 2-hop propagation
python main.py --model_type simple --dataset ml-100k

# 1-hop propagation (faster)
python main.py --model_type simple --dataset ml-100k --n_hops 1

# 2-hop with custom eigenvalues (single n_eigen only)
python main.py --model_type simple --dataset ml-100k --n_eigen 64 --n_hops 2

# Dual filter mode with 2-hop for better performance
python main.py --model_type simple --dataset ml-100k --filter_mode dual --n_eigen 96 --n_hops 2

# Large dataset with simple model (1-hop for speed)
python main.py --model_type simple --dataset gowalla --n_eigen 128 --n_hops 1
```

### Basic Model (Fast, Simple)
```bash
# Default configuration with separate eigenvalues
python main.py --model_type basic --dataset ml-100k --u_n_eigen 32 --i_n_eigen 48

# Legacy mode (same eigenvalues for both)
python main.py --model_type basic --dataset ml-100k --n_eigen 50

# Large dataset with basic model
python main.py --model_type basic --dataset gowalla --u_n_eigen 64 --i_n_eigen 96
```

### Enhanced Model (Advanced, Full Features)
```bash
# Auto-adaptive with enhanced features
python main.py --model_type enhanced --dataset ml-100k --filter_design enhanced_basis

# Manual eigenvalues with advanced filter
python main.py --model_type enhanced --dataset ml-100k --u_n_eigen 48 --i_n_eigen 64 --filter_design multiscale

# High-performance configuration
python main.py --model_type enhanced --dataset gowalla --u_n_eigen 128 --i_n_eigen 256 --filter_design ensemble
```

## Model Comparison

### When to Use Simple Model:
- **Ultra-fast experiments**: Quickest iteration time
- **Baseline comparisons**: Minimal implementation baseline
- **Resource constraints**: Lowest memory and computation
- **Learning core concepts**: Understanding spectral filtering basics
- **Prototype development**: Quick proof-of-concept

### When to Use Basic Model:
- **Fast prototyping**: Quick experiments and baseline comparisons
- **Limited resources**: Lower memory and computational requirements
- **Simple datasets**: Small to medium datasets (< 10K users/items)
- **Educational purposes**: Understanding core spectral filtering concepts
- **Debugging**: Simpler architecture for troubleshooting

### When to Use Enhanced Model:
- **Production systems**: Maximum performance requirements
- **Large datasets**: Complex, sparse datasets (> 10K users/items)
- **Research**: Advanced filter designs and similarity techniques
- **Domain-specific optimization**: Specialized similarity measures
- **Scalability**: Auto-adaptive features for varying dataset sizes

## Dataset-Specific Recommendations

### ML-100K (Small, Dense Dataset)
```bash
# Simple model - fastest option
python main.py --model_type simple --dataset ml-100k \
    --n_eigen 64 --epochs 30

# Basic model - fast and sufficient
python main.py --model_type basic --dataset ml-100k \
    --u_n_eigen 32 --i_n_eigen 48 \
    --epochs 50 --lr 0.001

# Enhanced model - maximum performance
python main.py --model_type enhanced --dataset ml-100k \
    --u_n_eigen 48 --i_n_eigen 64 \
    --filter_design enhanced_basis \
    --similarity_type cosine \
    --similarity_threshold 0.01
```

### LastFM (Music Domain)
```bash
# Simple model
python main.py --model_type simple --dataset lastfm \
    --n_eigen 96 --filter_mode dual

# Basic model
python main.py --model_type basic --dataset lastfm \
    --u_n_eigen 48 --i_n_eigen 64 \
    --epochs 40

# Enhanced model with music-optimized settings
python main.py --model_type enhanced --dataset lastfm \
    --u_n_eigen 64 --i_n_eigen 96 \
    --filter_design enhanced_basis \
    --similarity_type cosine \
    --similarity_threshold 0.005
```

### Gowalla (Location Data)
```bash
# Simple model - good baseline
python main.py --model_type simple --dataset gowalla \
    --n_eigen 128 --filter_mode dual

# Basic model - good baseline
python main.py --model_type basic --dataset gowalla \
    --u_n_eigen 64 --i_n_eigen 96 \
    --epochs 30

# Enhanced model - location-optimized
python main.py --model_type enhanced --dataset gowalla \
    --u_n_eigen 128 --i_n_eigen 256 \
    --filter_design multiscale \
    --similarity_type cosine \
    --similarity_threshold 0.001
```

## Performance vs Complexity Trade-offs

### Simple Model Characteristics:
- **Parameters**: ~20-50 (minimal overhead)
- **Training Time**: Fastest (3-5x faster than basic)
- **Memory Usage**: Lowest (30-50% of basic model)
- **Performance**: Good baseline (usually 80-90% of enhanced performance)
- **Suitable for**: Any dataset, especially for quick experiments

### Basic Model Characteristics:
- **Parameters**: ~100-500 (depending on eigenvalue counts)
- **Training Time**: Fast (2-5x faster than enhanced)
- **Memory Usage**: Low (50-70% of enhanced model)
- **Performance**: Good baseline (usually 85-95% of enhanced performance)
- **Suitable for**: Datasets < 50K users/items

### Enhanced Model Characteristics:
- **Parameters**: ~500-2000+ (depending on filter design)
- **Training Time**: Moderate to slow (comprehensive features)
- **Memory Usage**: Higher (caching, advanced processing)
- **Performance**: Maximum (state-of-the-art results)
- **Suitable for**: Any dataset, especially large/complex ones

## Eigenvalue Configuration Comparison

### Simple Model Eigenvalue Handling:
```bash
# Single eigenvalue count for simplified processing
python main.py --model_type simple --n_eigen 128

# Uses single eigendecomposition approach (like GF-CF)
# Fastest eigenvalue processing
# Good default values (128)
# Note: u_n_eigen/i_n_eigen not used, only n_eigen
```

### Basic Model Eigenvalue Handling:
```bash
# Simple, direct eigenvalue specification
python main.py --model_type basic --u_n_eigen 32 --i_n_eigen 48

# Uses basic heuristics for eigenvalue selection
# No auto-adaptive features
# Fast eigendecomposition
```

### Enhanced Model Eigenvalue Handling:
```bash
# Auto-adaptive based on dataset characteristics
python main.py --model_type enhanced --dataset ml-100k

# Manual with advanced similarity processing
python main.py --model_type enhanced --u_n_eigen 48 --i_n_eigen 64

# Intelligent caching and optimization
# Similarity-aware eigendecomposition
# Advanced threshold management
```

## Migration Guide

### From Simple to Basic:
```bash
# Step 1: Test with simple model
python main.py --model_type simple --dataset ml-100k --n_eigen 64

# Step 2: Convert to basic with separate eigenvalues
python main.py --model_type basic --dataset ml-100k --u_n_eigen 32 --i_n_eigen 48

# Step 3: Fine-tune eigenvalue allocation
python main.py --model_type basic --dataset ml-100k --u_n_eigen 48 --i_n_eigen 64
```

### From Basic to Enhanced:
```bash
# Step 1: Test with same eigenvalue configuration
python main.py --model_type basic --dataset ml-100k --u_n_eigen 32 --i_n_eigen 48
python main.py --model_type enhanced --dataset ml-100k --u_n_eigen 32 --i_n_eigen 48 --filter_design original

# Step 2: Enable enhanced features gradually
python main.py --model_type enhanced --dataset ml-100k --u_n_eigen 32 --i_n_eigen 48 --filter_design basis

# Step 3: Use auto-adaptive features
python main.py --model_type enhanced --dataset ml-100k --filter_design enhanced_basis

# Step 4: Optimize for your dataset
python main.py --model_type enhanced --dataset ml-100k --u_n_eigen 48 --i_n_eigen 64 --filter_design multiscale
```

## Debugging and Development

### Development Workflow:
```bash
# 1. Start with simple model for quickest iteration
python main.py --model_type simple --dataset ml-100k --n_eigen 32 --epochs 10

# 2. Move to basic for more control
python main.py --model_type basic --dataset ml-100k --u_n_eigen 24 --i_n_eigen 32 --epochs 20

# 3. Scale to enhanced for full features
python main.py --model_type enhanced --dataset ml-100k --filter_design basis --epochs 30

# 4. Optimize with advanced features
python main.py --model_type enhanced --dataset ml-100k --filter_design enhanced_basis --epochs 50
```

### Troubleshooting:
```bash
# Memory issues? Use simple model
python main.py --model_type simple --dataset large_dataset --n_eigen 64

# Need speed? Use simple model with dual filters
python main.py --model_type simple --dataset any_dataset --filter_mode dual

# Need maximum performance? Use enhanced
python main.py --model_type enhanced --dataset any_dataset --filter_design ensemble

# Debugging filters? Use simple with verbose output
python main.py --model_type simple --dataset ml-100k --verbose 1
```

## Performance Benchmarks (Approximate)

### ML-100K Results:
```
Simple Model:   NDCG@20 ≈ 0.370-0.380 (very fast training)
Basic Model:    NDCG@20 ≈ 0.375-0.385 (fast training)
Enhanced Model: NDCG@20 ≈ 0.385-0.395 (with optimization)
```

### Training Speed Comparison:
```
Simple Model:   ~15-30 seconds (ML-100K)
Basic Model:    ~30-60 seconds (ML-100K)
Enhanced Model: ~60-120 seconds (ML-100K, depending on filter design)
```

### Memory Usage:
```
Simple Model:   ~50-100 MB (ML-100K)
Basic Model:    ~100-200 MB (ML-100K)
Enhanced Model: ~200-400 MB (ML-100K, with caching)
```

## Advanced Usage Patterns

### Hyperparameter Search:
```bash
# Quick search with simple model
for n in 32 64 128; do
  python main.py --model_type simple --dataset ml-100k --n_eigen $n --epochs 15
done

# Medium search with basic model
for u in 24 32 48; do
  for i in 32 48 64; do
    python main.py --model_type basic --dataset ml-100k --u_n_eigen $u --i_n_eigen $i --epochs 20
  done
done

# Detailed search with enhanced model
python hyperparam_search.py --model_type enhanced --dataset ml-100k --filter_design enhanced_basis
```

### Cross-Model Validation:
```bash
# Train on simple, validate approach
python main.py --model_type simple --dataset ml-100k --n_eigen 64

# Scale up to basic for validation
python main.py --model_type basic --dataset ml-100k --u_n_eigen 32 --i_n_eigen 48

# Scale up to enhanced for final results
python main.py --model_type enhanced --dataset ml-100k --u_n_eigen 48 --i_n_eigen 64 --filter_design enhanced_basis
```

## Model Selection Guide

### Choose Simple Model When:
- You need the fastest possible iteration time
- You're doing initial experiments or prototyping
- You want to understand the core spectral filtering concept
- You have very limited computational resources
- You need a minimal baseline for comparison

### Choose Basic Model When:
- You want separate user/item eigenvalue control
- You need faster training than enhanced but more control than simple
- You're working with small to medium datasets
- You want a good balance of speed and performance
- You're debugging eigenvalue allocation strategies

### Choose Enhanced Model When:
- You need maximum performance
- You're working with large, complex datasets
- You want advanced similarity-aware processing
- You need domain-specific optimizations
- You're doing production deployments

## Best Practices

1. **Start Simple**: Always begin with the simple model to establish baseline performance
2. **Scale Gradually**: Move from simple → basic → enhanced as needed
3. **Match Model to Dataset**: Use simple for small datasets, enhanced for large ones
4. **Optimize Incrementally**: Don't jump straight to complex configurations
5. **Monitor Resources**: Keep an eye on memory usage and training time
6. **Validate Improvements**: Ensure each model upgrade actually improves performance

##### BEST HYPERPARAMETERS

FOR GOWALLA (Simple Model):
```bash
python main.py \
    --model_type simple \
    --dataset gowalla \
    --n_eigen 128 \
    --filter_mode dual \
    --lr 0.001 \
    --decay 0.01 \
    --epochs 30 \
    --patience 8
```

FOR GOWALLA (Basic Model):
```bash
python main.py \
    --model_type basic \
    --dataset gowalla \
    --lr 0.001 \
    --decay 0.01 \
    --u_n_eigen 230 \
    --i_n_eigen 180 \
    --filter ui \
    --filter_design enhanced_basis \
    --init_filter smooth \
    --epochs 50 \
    --patience 10
```

#!/bin/bash

# =============================================================================
# POLYNOMIAL FILTER USAGE EXAMPLES
# All existing filter designs continue to work exactly as before!
# =============================================================================

echo "🔢 POLYNOMIAL FILTER EXAMPLES FOR ENHANCED MODEL"
echo "All existing filters (enhanced_basis, multiscale, etc.) still work!"
echo ""

# =============================================================================
# 1. CHEBYSHEV POLYNOMIAL FILTER (Most Common)
# =============================================================================

echo "1️⃣ Chebyshev Polynomial Filter:"
python main.py \
    --model_type enhanced \
    --dataset ml-100k \
    --filter_design chebyshev \
    --u_n_eigen 48 \
    --i_n_eigen 64 \
    --init_filter smooth \
    --lr 0.001 \
    --decay 0.01

# =============================================================================
# 2. JACOBI POLYNOMIAL FILTER (Learnable Parameters)
# =============================================================================

echo "2️⃣ Jacobi Polynomial Filter with Learnable α, β:"
python main.py \
    --model_type enhanced \
    --dataset ml-100k \
    --filter_design jacobi \
    --polynomial_alpha 0.5 \
    --polynomial_beta -0.5 \
    --u_n_eigen 48 \
    --i_n_eigen 64 \
    --init_filter golden_036 \
    --lr 0.001 \
    --decay 0.01

# =============================================================================
# 3. GENERIC POLYNOMIAL FILTER (Specify Type)
# =============================================================================

echo "3️⃣ Generic Polynomial Filter (Laguerre):"
python main.py \
    --model_type enhanced \
    --dataset ml-100k \
    --filter_design polynomial \
    --polynomial_type laguerre \
    --polynomial_alpha 1.0 \
    --u_n_eigen 48 \
    --i_n_eigen 64 \
    --init_filter butterworth \
    --lr 0.001 \
    --decay 0.01

# =============================================================================
# 4. ADAPTIVE POLYNOMIAL FILTER (Learns Best Combination)
# =============================================================================

echo "4️⃣ Adaptive Polynomial Filter:"
python main.py \
    --model_type enhanced \
    --dataset ml-100k \
    --filter_design adaptive_polynomial \
    --adaptive_polynomial_types chebyshev legendre jacobi hermite \
    --u_n_eigen 48 \
    --i_n_eigen 64 \
    --init_filter smooth \
    --lr 0.001 \
    --decay 0.01

# =============================================================================
# 5. LEGENDRE POLYNOMIAL FILTER
# =============================================================================

echo "5️⃣ Legendre Polynomial Filter:"
python main.py \
    --model_type enhanced \
    --dataset gowalla \
    --filter_design legendre \
    --u_n_eigen 128 \
    --i_n_eigen 256 \
    --init_filter smooth \
    --lr 0.001 \
    --decay 0.01

# =============================================================================
# 6. THREE-VIEW WITH POLYNOMIAL FILTERS
# =============================================================================

echo "6️⃣ Three-View Model with Polynomial Filters:"
python main.py \
    --model_type enhanced \
    --dataset ml-100k \
    --filter uib \
    --filter_design chebyshev \
    --u_n_eigen 48 \
    --i_n_eigen 64 \
    --b_n_eigen 80 \
    --init_filter smooth \
    --lr 0.001 \
    --decay 0.01

# =============================================================================
# 7. EXISTING FILTERS STILL WORK (NO CHANGES NEEDED)
# =============================================================================

echo "7️⃣ Existing Enhanced Basis Filter (unchanged):"
python main.py \
    --model_type enhanced \
    --dataset ml-100k \
    --filter_design enhanced_basis \
    --u_n_eigen 48 \
    --i_n_eigen 64 \
    --init_filter smooth \
    --lr 0.001 \
    --decay 0.01

echo "8️⃣ Existing Ensemble Filter (unchanged):"
python main.py \
    --model_type enhanced \
    --dataset gowalla \
    --filter_design ensemble \
    --u_n_eigen 128 \
    --i_n_eigen 256 \
    --lr 0.001 \
    --decay 0.01

# =============================================================================
# PYTHON USAGE EXAMPLES
# =============================================================================

cat << 'EOF'

🐍 PYTHON API USAGE EXAMPLES:

```python
import torch
from model_enhanced import UniversalSpectralCF

# Example 1: Chebyshev Filter
config = {
    'dataset': 'ml-100k',
    'filter_design': 'chebyshev',
    'u_n_eigen': 48,
    'i_n_eigen': 64,
    'init_filter': 'smooth',
    'filter_order': 6
}
model = UniversalSpectralCF(adj_matrix, config)

# Example 2: Jacobi Filter with Custom Parameters
config = {
    'dataset': 'ml-100k',
    'filter_design': 'jacobi',
    'polynomial_params': {'alpha': 0.5, 'beta': -0.5},
    'u_n_eigen': 48,
    'i_n_eigen': 64,
    'init_filter': 'golden_036'
}
model = UniversalSpectralCF(adj_matrix, config)

# Example 3: Generic Polynomial Filter
config = {
    'dataset': 'gowalla',
    'filter_design': 'polynomial',
    'polynomial_type': 'hermite',
    'u_n_eigen': 128,
    'i_n_eigen': 256,
    'init_filter': 'smooth'
}
model = UniversalSpectralCF(adj_matrix, config)

# Example 4: Adaptive Polynomial Filter
config = {
    'dataset': 'ml-100k',
    'filter_design': 'adaptive_polynomial',
    'polynomial_params': {
        'types': ['chebyshev', 'legendre', 'jacobi', 'laguerre']
    },
    'u_n_eigen': 48,
    'i_n_eigen': 64
}
model = UniversalSpectralCF(adj_matrix, config)

# Check what polynomial types were learned as most important
if hasattr(model.user_filter, 'get_type_weights'):
    print("Learned polynomial type weights:", model.user_filter.get_type_weights())
```

🔍 DEBUGGING POLYNOMIAL FILTERS:

```python
# After training, debug polynomial filter learning:
model.debug_filter_learning()

# For Jacobi filters, check learned parameters:
if hasattr(model.user_filter, 'alpha'):
    print(f"Learned α: {model.user_filter.alpha.item():.4f}")
    print(f"Learned β: {model.user_filter.beta.item():.4f}")

# For adaptive polynomial filters, check type importance:
if hasattr(model.user_filter, 'get_type_weights'):
    weights = model.user_filter.get_type_weights()
    print("Polynomial type importance:")
    for poly_type, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {poly_type}: {weight:.3f}")
```

🎯 FILTER DESIGN COMPARISON:

All these filter designs are now available:

EXISTING (PRESERVED):
- 'original'              # Original universal filter
- 'basis'                # Basis filter combination  
- 'enhanced_basis'       # Enhanced basis (default)
- 'adaptive_golden'      # Golden ratio adaptive
- 'multiscale'           # Multi-scale filtering
- 'ensemble'             # Ensemble of multiple filters
- 'band_stop'            # Band-stop frequency filter
- 'adaptive_band_stop'   # Adaptive band-stop
- 'parametric_multi_band' # Parametric multi-band
- 'harmonic'             # Harmonic series filter

NEW POLYNOMIAL FILTERS:
- 'polynomial'           # Generic (specify polynomial_type)
- 'chebyshev'           # Chebyshev polynomials
- 'jacobi'              # Jacobi polynomials (learnable α, β)
- 'legendre'            # Legendre polynomials
- 'adaptive_polynomial' # Learns best polynomial combination

📊 EXPECTED PERFORMANCE:

Based on spectral filtering theory:
- Chebyshev: Excellent for most datasets (proven effectiveness)
- Jacobi: Flexible, can adapt α,β parameters to data characteristics
- Legendre: Good alternative to Chebyshev, orthogonal on [-1,1]
- Adaptive: Automatically finds best polynomial combination

🔬 RESEARCH APPLICATIONS:

1. Compare polynomial families on your dataset
2. Study how α,β parameters evolve during training
3. Analyze which polynomial types adaptive filter prefers
4. Investigate spectral properties of different bases

EOF

echo ""
echo "✅ All examples preserve existing functionality!"
echo "🔢 New polynomial filters are additions, not replacements!"
echo "🚀 Use your existing commands - they all still work!"
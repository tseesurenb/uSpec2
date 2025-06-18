# Search Tools for Learnable Spectral CF

This directory contains search tools to optimize the learnable spectral collaborative filtering model.

## Search Files

### 🎯 `search_filter_patterns.py` - **MOST IMPORTANT**
**Purpose**: Find optimal filter initialization patterns for different datasets
**Why crucial**: Amazon-book, Yelp2018, Gowalla have clustered eigenvalues that break standard initializations

```bash
# Find best item filter pattern for Amazon-book
python search_filter_patterns.py --dataset amazon-book --view item --eigenvalues 600

# Find best user filter pattern
python search_filter_patterns.py --dataset amazon-book --view user --eigenvalues 130
```

**Output**: Saves best patterns as `.npy` files for learnable filter initialization

### 🔧 `search_eigenvalues.py`
**Purpose**: Find optimal number of eigenvalues per view
**Use**: After fixing filter patterns

```bash
python search_eigenvalues.py --dataset ml-100k --filter uib --filter_type spectral_basis --loss mse
```

### 📈 `search_learning_rates.py`
**Purpose**: Find optimal learning rates and weight decay per view
**Use**: Final tuning step after eigenvalues and patterns are set

```bash
python search_learning_rates.py --dataset ml-100k --filter uib --filter_type spectral_basis --loss mse --u 10 --i 300 --b 50
```

## Current Issues

**Problem**: Amazon-book, Yelp2018, Gowalla all perform poorly because:
1. **Clustered eigenvalues**: [0.260, 0.261, 0.262...] instead of spread-out values
2. **Wrong initialization patterns**: "smooth", "sharp" designed for spread-out eigenvalues
3. **Filter mismatch**: Patterns don't match the actual data structure

**Solution**: Use `search_filter_patterns.py` to find the right initialization patterns first!

## Recommended Workflow

1. **First**: Run `search_filter_patterns.py` for each dataset/view
2. **Second**: Update learnable filter initializations with discovered patterns  
3. **Third**: Run `search_eigenvalues.py` to optimize eigenvalue counts
4. **Finally**: Run `search_learning_rates.py` for final tuning

## Priority

**Focus on filter patterns first** - this is likely the root cause of poor performance on larger datasets.
# SpecLearn v2: Enhanced with DySimGCF Innovations

This directory contains an enhanced version of the learnable spectral collaborative filtering model that incorporates key innovations from the DySimGCF paper.

## Key Enhancements

### 1. Improved Raw Item Similarity Model (`improved_raw_model.py`)

This new model combines the insights from your pattern search with DySimGCF's innovations:

- **Raw Matrix Multiplication**: Uses simple and efficient `R.T @ R` (raw adjacency) for item similarity
- **Symmetric Softmax Normalization**: **Replaces** degree-based normalization (D^{-0.5} R D^{-0.5}) with edge-weight-based softmax normalization
- **Temperature-based Similarity Amplification**: Uses learnable temperature to control similarity emphasis
- **Top-K Similarity Selection**: Retains only the most relevant item-item connections to reduce noise
- **No Spectral Filtering**: Pure collaborative signal through raw edge weight propagation

### 2. Model Types

**Learnable Spectral** (`--model_type learnable`):
- Original spectral filtering approach with your discovered optimal patterns
- Uses eigendecomposition and learnable filters

**Improved Raw** (`--model_type improved_raw`):
- DySimGCF-inspired symmetric attention on item similarities
- Temperature scaling for similarity amplification
- Much faster (no eigendecomposition needed)

## Usage Examples

### Test Improved Raw Model on Gowalla (No Degree Normalization)
```bash
python main.py --dataset gowalla --model_type improved_raw --temperature 1.5 --top_k_items 50 --epochs 50
```

### Compare with Original Learnable Model
```bash
# Original learnable model
python main.py --dataset gowalla --model_type learnable --filter i --filter_type spectral_basis --item_init exp_decay --i 600

# Improved raw model
python main.py --dataset gowalla --model_type improved_raw --temperature 2.0 --top_k_items 80
```

### Yelp2018 with Optimal Settings
```bash
python main.py --dataset yelp2018 --model_type improved_raw --temperature 1.2 --top_k_items 40
```

## Key Parameters for Improved Raw Model

- `--temperature`: Controls similarity amplification (1.0-3.0, higher = more emphasis on strong similarities)
- `--top_k_items`: Number of most similar items to retain per item (20-100)

## Expected Benefits

1. **Performance**: Should match or exceed raw two-hop performance from v1
2. **Speed**: Much faster than spectral filtering (no eigendecomposition)
3. **Adaptability**: Temperature learning allows dataset-specific optimization
4. **Noise Reduction**: Top-K selection removes weak/noisy connections

## Architecture Comparison

**v1 Raw Two-hop**:
```
user_profile @ norm_adj.T @ norm_adj  # Uses degree normalization
```

**v2 Improved Raw**:
```
user_profile @ attention_weighted_item_similarity
where similarity = R.T @ R  # Raw adjacency
attention = exp(similarity/T) / sqrt(sum_i * sum_j)  # Symmetric softmax (like D^{-0.5} but with edge weights)
```

## Files Modified from v1

- `config.py`: Added new model type and DySimGCF parameters
- `main.py`: Added model selection and enhanced logging
- `improved_raw_model.py`: New model with DySimGCF innovations

## Preserving v1

The original `speclearn_v1` directory remains completely intact for comparison and fallback.
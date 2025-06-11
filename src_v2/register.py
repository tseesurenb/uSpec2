'''
Created on June 7, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering
Enhanced with user-specific model selection

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''
import world
import dataloader

# Dataset loading
if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()
elif world.dataset == 'ml-100k':
    dataset = dataloader.ML100K()
else:
    raise ValueError(f"Unknown dataset: {world.dataset}")

# Model selection based on configuration - UPDATED WITH USER-SPECIFIC MODEL
model_type = world.config.get('model_type', 'enhanced')

if model_type == 'simple':
    import simple_model
    MODELS = {'uspec': simple_model.SimpleUniversalSpectralCF}
    print("⚡ Using Simple Universal Spectral CF (simple_model.py)")
    print(f"   └─ Minimal implementation with single eigendecomposition")
    print(f"   └─ Fast training, learnable spectral filters")
    print(f"   └─ Filter mode: {world.config.get('filter_mode', 'single')}")
    
elif model_type == 'basic':
    import model
    MODELS = {'uspec': model.UniversalSpectralCF}
    print("🔧 Using Basic Universal Spectral CF (model.py)")
    print(f"   └─ Simple eigendecomposition with separate u_n_eigen/i_n_eigen")
    print(f"   └─ Fast training, minimal complexity")
    
elif model_type == 'enhanced':
    import model_enhanced
    MODELS = {'uspec': model_enhanced.UniversalSpectralCF}
    print("🚀 Using Enhanced Universal Spectral CF (model_enhanced.py)")
    print(f"   └─ DySimGCF-style similarity-aware Laplacian")
    print(f"   └─ Advanced filter designs and caching")
    print(f"   └─ Adaptive eigenvalue calculation")

elif model_type == 'user_specific':
    import model_personalized
    MODELS = {'uspec': model_personalized.UserSpecificUniversalSpectralCF}
    print("🎯 Using User-Specific Universal Spectral CF (model_user_specific.py)")
    print(f"   └─ Personalized spectral filter parameters for each user")
    print(f"   └─ Shared base: {world.config.get('shared_base', True)}")
    print(f"   └─ Personalization dim: {world.config.get('personalization_dim', 8)}")
    print(f"   └─ Cold start strategy: {world.config.get('cold_start_strategy', 'average')}")
    
else:
    raise ValueError(f"Unknown model_type: {model_type}. Choose 'simple', 'basic', 'enhanced', or 'user_specific'")

# Display configuration info
if world.config['verbose'] > 0:
    print(f"\n📊 Dataset Configuration:")
    print(f"   └─ Dataset: {world.dataset}")
    print(f"   └─ Users: {dataset.n_users:,}, Items: {dataset.m_items:,}")
    print(f"   └─ Training: {dataset.trainDataSize:,}, Validation: {dataset.valDataSize:,}")
    
    print(f"\n⚙️  Model Configuration:")
    print(f"   └─ Model Type: {model_type}")
    
    # Eigenvalue configuration
    u_n_eigen = world.config.get('u_n_eigen', 0)
    i_n_eigen = world.config.get('i_n_eigen', 0)
    n_eigen = world.config.get('n_eigen', 0)
    
    if u_n_eigen > 0 and i_n_eigen > 0:
        print(f"   └─ User Eigenvalues: {u_n_eigen}")
        print(f"   └─ Item Eigenvalues: {i_n_eigen}")
        print(f"   └─ Eigenvalue Ratio (i/u): {i_n_eigen/u_n_eigen:.2f}")
    elif n_eigen > 0:
        print(f"   └─ Eigenvalues (legacy): {n_eigen}")
    else:
        if model_type == 'simple':
            print(f"   └─ Eigenvalues: Default (128 each)")
        else:
            print(f"   └─ Eigenvalues: Auto-adaptive")
    
    # Model-specific configuration
    if model_type == 'simple':
        print(f"   └─ Filter Mode: {world.config.get('filter_mode', 'single')}")
    elif model_type == 'enhanced':
        print(f"   └─ Filter Design: {world.config.get('filter_design', 'enhanced_basis')}")
        print(f"   └─ Similarity Type: {world.config.get('similarity_type', 'cosine')}")
        print(f"   └─ Similarity Threshold: {world.config.get('similarity_threshold', 0.01)}")
    elif model_type == 'user_specific':
        print(f"   └─ Shared Base: {world.config.get('shared_base', True)}")
        print(f"   └─ Personalization Dim: {world.config.get('personalization_dim', 8)}")
        print(f"   └─ Cold Start Strategy: {world.config.get('cold_start_strategy', 'average')}")
        print(f"   └─ Similarity Type: {world.config.get('similarity_type', 'cosine')}")
        print(f"   └─ Similarity Threshold: {world.config.get('similarity_threshold', 0.01)}")
    
    print(f"   └─ Filter Type: {world.config['filter']}")
    print(f"   └─ Filter Order: {world.config['filter_order']}")
    print(f"   └─ Device: {world.device}")

# Legacy compatibility checks
if world.config.get('use_laplacian', False):
    print(f"⚠️  Note: use_laplacian flag detected. Enhanced model uses similarity-aware Laplacian by default.")

if world.config.get('use_similarity_norm', False):
    print(f"⚠️  Note: use_similarity_norm flag detected. Enhanced model uses advanced similarity processing by default.")
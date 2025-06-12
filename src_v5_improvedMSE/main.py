'''
Created on June 12, 2025
Main file for Improved Universal Spectral CF with MSE loss
Optimized for matrix-based collaborative filtering

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import world
import utils
import procedure
import time
from register import dataset, MODELS
import warnings
warnings.filterwarnings("ignore")

# Set random seed
utils.set_seed(world.seed)

print(f"Improved Universal Spectral CF: {world.config['dataset']} dataset")
print(f"Filter: {world.config['filter']}, Device: {world.device}")
print(f"Training mode: MSE Loss with {world.config.get('samples', 1)} negative samples")

# Create model
print(f"Creating improved spectral model...")
model_start = time.time()
adj_mat = dataset.UserItemNet.tolil()

# Use the improved model
ImprovedUniversalSpectralCF = MODELS['uspec']
model = ImprovedUniversalSpectralCF(adj_mat, world.config)
print(f"Model created in {time.time() - model_start:.2f}s")

# Dataset info
print(f"Dataset: {dataset.n_users:,} users, {dataset.m_items:,} items")
print(f"Training: {dataset.trainDataSize:,} interactions")

# Display cache info for transparency
try:
    from model import get_cache_info
    get_cache_info(world.dataset)
except:
    pass

# Training with improved MSE procedure
trained_model, final_results = procedure.train_and_evaluate_spectral(dataset, model, world.config)

# Print final performance comparison
print(f"\n🏆 \033[92mPerformance Summary (MSE Loss):\033[0m")
print(f"🎯 Target NDCG@20: 0.45 (competitive baseline)")
print(f"📈 Achieved NDCG@20: {final_results['ndcg'][0]:.4f}")

# Calculate improvement over a reasonable baseline
baseline_ndcg = 0.35  # Conservative baseline for spectral methods
improvement = (final_results['ndcg'][0] - baseline_ndcg) / baseline_ndcg * 100

print(f"📊 Improvement over baseline: {improvement:+.1f}%")

# Performance assessment
if final_results['ndcg'][0] >= 0.42:
    print(f"✅ \033[92mExcellent!\033[0m Spectral model with MSE achieves competitive performance!")
elif final_results['ndcg'][0] >= 0.38:
    print(f"⚡ \033[93mGood performance!\033[0m MSE loss works well for spectral models.")
elif final_results['ndcg'][0] >= 0.32:
    print(f"📈 \033[96mSolid results!\033[0m Room for hyperparameter tuning.")
else:
    print(f"🔧 \033[91mNeed improvements.\033[0m Consider different configurations.")

# Performance insights
print(f"\n💡 \033[95mMSE Loss Insights:\033[0m")
if final_results['recall'][0] > final_results['precision'][0] * 1.5:
    print("📊 High recall suggests good coverage - MSE reconstructs interaction patterns well")
elif final_results['precision'][0] > 0.25:
    print("🎯 High precision suggests good ranking - spectral filtering is effective")

print(f"🔄 \033[94mRecommendation:\033[0m MSE loss is well-suited for spectral collaborative filtering")
print(f"   └─ Matrix reconstruction aligns with spectral filter objectives")
print(f"   └─ Simpler optimization compared to ranking-based losses")

# Suggest next steps based on performance
if final_results['ndcg'][0] < 0.38:
    print(f"\n🔧 \033[93mSuggested improvements:\033[0m")
    print(f"   └─ Try --preset gowalla_optimized for higher learning rate")
    print(f"   └─ Increase negative samples: --samples 100")
    print(f"   └─ Experiment with filter combinations: --filter uib")
    print(f"   └─ Adjust eigenvalue counts: --u_n_eigen 50 --i_n_eigen 100")
'''
Created on June 12, 2025
Main file for Improved Universal Spectral CF with BPR loss
Based on DySimGCF's successful methodology

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
print(f"Training mode: BPR Loss with {world.config.get('samples', 1)} negative samples")

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

# Training with improved procedure
trained_model, final_results = procedure.train_and_evaluate_spectral(dataset, model, world.config)

# Print final performance comparison
print(f"\n🏆 \033[92mPerformance Summary:\033[0m")
print(f"🎯 Target NDCG@20: 0.45 (DySimGCF baseline)")
print(f"📈 Achieved NDCG@20: {final_results['ndcg'][0]:.4f}")
improvement = (final_results['ndcg'][0] - 0.388) / 0.388 * 100
print(f"📊 Improvement over original: {improvement:+.1f}%")

if final_results['ndcg'][0] >= 0.42:
    print(f"✅ \033[92mSuccess!\033[0m Spectral model is competitive with GCN methods!")
elif final_results['ndcg'][0] >= 0.40:
    print(f"⚡ \033[93mGood progress!\033[0m Getting close to the target.")
else:
    print(f"🔧 \033[91mNeed more improvements.\033[0m Consider adjusting hyperparameters.")
"""
Minimal main script for clean spectral CF
Clean, focused implementation with MSE loss only
"""
import torch
import numpy as np
import time
import os

from config import parse_args, get_config
from spectral_cf import SpectralCF, MSE_train
from laplacian_cf import LaplacianCF
from dataloader import Loader


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def test_one_batch(X):
    """Simple test function"""
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    
    # Simple metrics calculation
    recall_20 = []
    precision_20 = []
    ndcg_20 = []
    
    for i, items in enumerate(sorted_items):
        ground_true = groundTrue[i]
        if len(ground_true) == 0:
            recall_20.append(0)
            precision_20.append(0) 
            ndcg_20.append(0)
            continue
            
        # Top-20 recommendations
        top_20 = items[:20]
        hits = len(set(top_20) & set(ground_true))
        
        # Recall@20
        recall_20.append(hits / len(ground_true))
        
        # Precision@20
        precision_20.append(hits / 20)
        
        # NDCG@20 (simplified)
        dcg = sum([1/np.log2(j+2) for j, item in enumerate(top_20) if item in ground_true])
        idcg = sum([1/np.log2(j+2) for j in range(min(len(ground_true), 20))])
        ndcg_20.append(dcg / idcg if idcg > 0 else 0)
    
    return {
        'recall': np.mean(recall_20),
        'precision': np.mean(precision_20),
        'ndcg': np.mean(ndcg_20)
    }


def evaluate(dataset, model):
    """Evaluation function"""
    testDict = dataset.testDict
    users = list(testDict.keys())
    if len(users) == 0:
        return {'recall': 0, 'precision': 0, 'ndcg': 0}
    
    batch_size = 500
    all_results = []
    
    for i in range(0, len(users), batch_size):
        batch_users = users[i:i + batch_size]
        
        # Get ground truth
        groundTrue = [testDict[u] for u in batch_users]
        
        # Get model predictions
        ratings = model.getUsersRating(batch_users)
        
        # Exclude training items
        for j, user in enumerate(batch_users):
            train_items = dataset.allPos[user]
            ratings[j, train_items] = -np.inf
        
        # Get top-k items
        _, top_items = torch.topk(torch.tensor(ratings), k=20, dim=1)
        
        # Evaluate this batch
        result = test_one_batch((top_items, groundTrue))
        all_results.append(result)
    
    # Average results
    avg_recall = np.mean([r['recall'] for r in all_results])
    avg_precision = np.mean([r['precision'] for r in all_results]) 
    avg_ndcg = np.mean([r['ndcg'] for r in all_results])
    
    return {
        'recall': avg_recall,
        'precision': avg_precision, 
        'ndcg': avg_ndcg
    }


def main():
    # Parse arguments
    args = parse_args()
    config = get_config(args)
    
    # Set seed
    set_seed(config['seed'])
    
    # Print config
    if config['verbose'] > 0:
        print("=" * 50)
        print("Spectral CF")
        print("=" * 50)
        print(f"Dataset: {config['dataset']}")
        print(f"Eigenvalues: u={config['u_eigen']}, i={config['i_eigen']}, b={config['b_eigen']}")
        print(f"Learning rates: u={config['user_lr']}, i={config['item_lr']}, b={config['bipartite_lr']}")
        print(f"Device: {config['device']}")
        print("=" * 50)
    
    # Load dataset
    dataset = Loader(config)
    
    # Create model based on type
    if config.get('use_laplacian', False):
        print(f"Using Laplacian-based spectral filtering ({config['laplacian_type']})")
        model = LaplacianCF(dataset.UserItemNet, config).to(config['device'])
    else:
        print("Using similarity-based spectral filtering")
        model = SpectralCF(dataset.UserItemNet, config).to(config['device'])
    
    # Create optimizer with per-view learning rates
    optimizer_groups = model.get_optimizer_groups()
    optimizer = torch.optim.Adam(optimizer_groups)
    
    # Training loop
    best_ndcg = 0
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = MSE_train(dataset, model, optimizer, config['batch_size'])
        
        # Evaluation
        if (epoch + 1) % config['eval_freq'] == 0:
            model.eval()
            with torch.no_grad():
                results = evaluate(dataset, model)
            
            ndcg = results['ndcg']
            recall = results['recall'] 
            precision = results['precision']
            
            print(f"Epoch {epoch+1}/{config['epochs']}")
            print(f"Loss: {train_loss:.4f}")
            print(f"Test NDCG@20: {ndcg:.4f} | Recall@20: {recall:.4f} | Precision@20: {precision:.4f}")
            
            # Track best
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                print("New best!")
    
    print(f"\nTraining completed!")
    print(f"Best NDCG@20: {best_ndcg:.4f}")


if __name__ == "__main__":
    main()
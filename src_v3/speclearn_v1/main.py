"""
Main training script for Learnable Spectral CF
Self-contained version
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message="Can't initialize NVML")

import torch
import numpy as np
import time
import os

# All imports from current directory
from config import parse_args, get_config
from learnable_model import SpectralCFLearnable
from dataloader import Loader
import utils


def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_optimizer(model, config):
    """Create optimizer with per-view hyperparameters"""
    param_groups = model.get_optimizer_groups()
    
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(param_groups)
    elif config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(param_groups)
    elif config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    return optimizer


def get_scheduler(optimizer, config):
    """Create learning rate scheduler"""
    if config['scheduler'] == 'none':
        return None
    elif config['scheduler'] == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=1e-6
        )
    elif config['scheduler'] == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.5
        )
    elif config['scheduler'] == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
    else:
        raise ValueError(f"Unknown scheduler: {config['scheduler']}")


def log_filter_response(model, epoch, save_dir):
    """Log filter responses for visualization"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping filter logging")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # User filter
    if hasattr(model, 'user_filter'):
        x, y = model.user_filter.get_filter_values()
        axes[0].plot(x, y, 'b-', linewidth=2)
        axes[0].set_title('User Filter Response')
        axes[0].set_xlabel('Eigenvalue')
        axes[0].set_ylabel('Filter Value')
        axes[0].grid(True, alpha=0.3)
    
    # Item filter
    if hasattr(model, 'item_filter'):
        x, y = model.item_filter.get_filter_values()
        axes[1].plot(x, y, 'g-', linewidth=2)
        axes[1].set_title('Item Filter Response')
        axes[1].set_xlabel('Eigenvalue')
        axes[1].set_ylabel('Filter Value')
        axes[1].grid(True, alpha=0.3)
    
    # Bipartite filter
    if hasattr(model, 'bipartite_filter'):
        x, y = model.bipartite_filter.get_filter_values()
        axes[2].plot(x, y, 'r-', linewidth=2)
        axes[2].set_title('Bipartite Filter Response')
        axes[2].set_xlabel('Eigenvalue')
        axes[2].set_ylabel('Filter Value')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'filters_epoch_{epoch}.png'))
    plt.close()


def main():
    # Parse arguments
    args = parse_args()
    config = get_config(args)
    
    # Set seed
    set_seed(config['seed'])
    
    # Print configuration
    if config['verbose'] > 0:
        print("\n" + "="*60)
        print("Learnable Spectral CF Configuration")
        print("="*60)
        print(f"Dataset: {config['dataset']}")
        print(f"Filter views: {config['filter']}")
        print(f"Filter type: {config['filter_type']} (order {config['filter_order']})")
        print(f"Eigenvalues: u={config['u_n_eigen']}, i={config['i_n_eigen']}, b={config['b_n_eigen']}")
        print(f"Device: {config['device']}")
        print("="*60 + "\n")
    
    # Load dataset
    dataset = Loader(config)
    
    # Create model
    model = SpectralCFLearnable(dataset.UserItemNet, config).to(config['device'])
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    # Create experiment directory
    exp_dir = f"experiments/{config['exp_name']}_{config['dataset']}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # Training loop
    best_ndcg = 0
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        if config['loss'] == 'bpr':
            train_loss = BPR_train_learnable(
                dataset, model, optimizer, 
                neg_ratio=config['neg_ratio'],
                batch_size=config['train_batch_size']
            )
        else:  # MSE
            train_loss = MSE_train_learnable(
                dataset, model, optimizer,
                batch_size=config['train_batch_size']
            )
        
        # Learning rate scheduling
        if scheduler and config['scheduler'] != 'plateau':
            scheduler.step()
        
        # Evaluation
        if (epoch + 1) % config['eval_freq'] == 0:
            model.eval()
            with torch.no_grad():
                if config['full_training']:
                    # No validation set, evaluate on test set (for monitoring only)
                    results = Test(dataset, model, epoch)
                    eval_name = "test"
                else:
                    # Use validation set for hyperparameter tuning
                    results = Test_val(dataset, model, epoch)
                    eval_name = "validation"
            
            ndcg = results['ndcg'][0]
            recall = results['recall'][0]
            precision = results['precision'][0]
            
            print(f"\nEpoch {epoch+1}/{config['epochs']}")
            print(f"Loss: {train_loss:.4f}")
            print(f"{eval_name.capitalize()} NDCG@20: {ndcg:.4f} | Recall@20: {recall:.4f} | Precision@20: {precision:.4f}")
            
            # Log learning rates
            for param_group in optimizer.param_groups:
                print(f"{param_group['name']} LR: {param_group['lr']:.6f}")
            
            # Log two-hop weight if used
            if hasattr(model, 'use_two_hop') and model.use_two_hop and model.dataset != 'amazon-book':
                print(f"Two-hop weight: {model.two_hop_weight.item():.4f}")
            
            # Plateau scheduler
            if scheduler and config['scheduler'] == 'plateau':
                scheduler.step(ndcg)
            
            # Log filter responses
            if config['log_filters']:
                log_filter_response(model, epoch+1, exp_dir)
            
            # Early stopping
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                patience_counter = 0
                
                # Save best model
                if config['save_model']:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ndcg': ndcg,
                        'config': config
                    }, os.path.join(exp_dir, 'best_model.pth'))
                    print("Saved best model!")
            else:
                patience_counter += 1
                
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print(f"\nTraining completed!")
    print(f"Best NDCG@20: {best_ndcg:.4f}")
    print(f"Results saved to: {exp_dir}")


def MSE_train_learnable(dataset, model, optimizer, batch_size=2048):
    """MSE training for learnable spectral model (faster, simpler)"""
    model.train()
    
    # Sample users for training
    n_users = dataset.n_users
    sample_size = min(batch_size, n_users)  # Don't exceed number of users
    users = np.random.choice(n_users, sample_size, replace=False)
    users = torch.tensor(users, dtype=torch.long).to(model.device)
    
    # Create target ratings (binary: 1 for positive interactions, 0 for others)
    batch_size = len(users)
    n_items = dataset.m_items
    target_ratings = torch.zeros(batch_size, n_items, device=model.device)
    
    # Fill in positive interactions
    for i, user in enumerate(users.cpu().numpy()):
        pos_items = dataset.allPos[user]
        if len(pos_items) > 0:
            target_ratings[i, pos_items] = 1.0
    
    # Forward pass
    optimizer.zero_grad()
    predicted_ratings = model(users)
    
    # MSE loss
    loss = torch.mean((predicted_ratings - target_ratings) ** 2)
    
    # Add regularization
    reg_loss = 0
    if hasattr(model, 'user_filter'):
        for param in model.user_filter.parameters():
            reg_loss += model.user_decay * torch.norm(param, 2)
    if hasattr(model, 'item_filter'):
        for param in model.item_filter.parameters():
            reg_loss += model.item_decay * torch.norm(param, 2)
    if hasattr(model, 'bipartite_filter'):
        for param in model.bipartite_filter.parameters():
            reg_loss += model.bipartite_decay * torch.norm(param, 2)
    
    total_loss = loss + reg_loss
    
    # Backward and optimize
    total_loss.backward()
    optimizer.step()
    
    return total_loss.cpu().item()


def BPR_train_learnable(dataset, model, optimizer, neg_ratio=1, batch_size=2048):
    """BPR training for learnable spectral model"""
    model.train()
    
    # Get positive samples
    S = utils.UniformSample_original(dataset, neg_ratio)
    users = torch.Tensor(S[:, 0]).long().to(model.device)
    posItems = torch.Tensor(S[:, 1]).long().to(model.device)
    negItems = torch.Tensor(S[:, 2]).long().to(model.device)
    
    # Shuffle
    n_batch = len(users) // batch_size + 1
    aver_loss = 0.
    
    for batch_i, (batch_users, batch_pos, batch_neg) in enumerate(
        utils.minibatch(users, posItems, negItems, batch_size=batch_size)
    ):
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        all_scores = model(batch_users)  # (batch_size, n_items)
        
        # Get scores for positive and negative items
        pos_scores = torch.gather(all_scores, 1, batch_pos.unsqueeze(1)).squeeze()
        neg_scores = torch.gather(all_scores, 1, batch_neg.unsqueeze(1)).squeeze()
        
        # BPR loss
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
        
        # Add regularization
        reg_loss = 0
        if hasattr(model, 'user_filter'):
            for param in model.user_filter.parameters():
                reg_loss += model.user_decay * torch.norm(param, 2)
        if hasattr(model, 'item_filter'):
            for param in model.item_filter.parameters():
                reg_loss += model.item_decay * torch.norm(param, 2)
        if hasattr(model, 'bipartite_filter'):
            for param in model.bipartite_filter.parameters():
                reg_loss += model.bipartite_decay * torch.norm(param, 2)
        
        loss = loss + reg_loss
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        aver_loss += loss.cpu().item()
    
    return aver_loss / n_batch


def test_one_batch(X):
    """Test one batch"""
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    topks = [20]  # Default topk
    for k in topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall), 
            'precision': np.array(pre), 
            'ndcg': np.array(ndcg)}


def Test_val(dataset, model, epoch, config=None):
    """Validation function"""
    u_batch_size = 500
    testDict = dataset.valDict
    max_K = 20
    
    results = {'precision': np.zeros(1),
               'recall': np.zeros(1),
               'ndcg': np.zeros(1)}
    
    users = list(testDict.keys())
    if len(users) == 0:
        return results
    
    users_list = []
    rating_list = []
    groundTrue_list = []
    
    # Process in batches
    for i in range(0, len(users), u_batch_size):
        batch_users = users[i:i + u_batch_size]
        allPos = dataset.getUserPosItems(batch_users)
        groundTrue = [testDict[u] for u in batch_users]
        
        # Get ratings from model
        rating = model.getUsersRating(batch_users)
        rating = torch.from_numpy(rating) if isinstance(rating, np.ndarray) else rating
        
        # Exclude training items
        exclude_index = []
        exclude_items = []
        for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i] * len(items))
            exclude_items.extend(items)
        rating[exclude_index, exclude_items] = -(1<<10)
        _, rating_K = torch.topk(rating, k=max_K)
        
        users_list.append(batch_users)
        rating_list.append(rating_K.cpu())
        groundTrue_list.append(groundTrue)
    
    # Test batches
    X = zip(rating_list, groundTrue_list)
    pre_results = []
    for x in X:
        pre_results.append(test_one_batch(x))
    
    # Aggregate results
    for result in pre_results:
        results['recall'] += result['recall']
        results['precision'] += result['precision']
        results['ndcg'] += result['ndcg']
    results['recall'] /= float(len(users))
    results['precision'] /= float(len(users))
    results['ndcg'] /= float(len(users))
    
    return results


def Test(dataset, model, epoch, config=None):
    """Test function"""
    u_batch_size = 500
    testDict = dataset.testDict
    max_K = 20
    
    results = {'precision': np.zeros(1),
               'recall': np.zeros(1),
               'ndcg': np.zeros(1)}
    
    users = list(testDict.keys())
    if len(users) == 0:
        return results
    
    users_list = []
    rating_list = []
    groundTrue_list = []
    
    # Process in batches
    for i in range(0, len(users), u_batch_size):
        batch_users = users[i:i + u_batch_size]
        allPos = dataset.getUserPosItems(batch_users)
        groundTrue = [testDict[u] for u in batch_users]
        
        # Get ratings from model
        rating = model.getUsersRating(batch_users)
        rating = torch.from_numpy(rating) if isinstance(rating, np.ndarray) else rating
        
        # Exclude training items
        exclude_index = []
        exclude_items = []
        for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i] * len(items))
            exclude_items.extend(items)
        rating[exclude_index, exclude_items] = -(1<<10)
        _, rating_K = torch.topk(rating, k=max_K)
        
        users_list.append(batch_users)
        rating_list.append(rating_K.cpu())
        groundTrue_list.append(groundTrue)
    
    # Test batches
    X = zip(rating_list, groundTrue_list)
    pre_results = []
    for x in X:
        pre_results.append(test_one_batch(x))
    
    # Aggregate results
    for result in pre_results:
        results['recall'] += result['recall']
        results['precision'] += result['precision']
        results['ndcg'] += result['ndcg']
    results['recall'] /= float(len(users))
    results['precision'] /= float(len(users))
    results['ndcg'] /= float(len(users))
    
    return results


if __name__ == "__main__":
    main()
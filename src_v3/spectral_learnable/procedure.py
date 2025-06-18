'''
Created on June 12, 2025
Simplified training procedure
Minimalist approach similar to GF-CF

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import world
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from time import time
from tqdm import tqdm

class MSELoss:
    """Simple MSE Loss"""
    def __init__(self, model, config):
        self.model = model
        base_lr = config['lr']
        weight_decay = config['decay']
        
        # Check if model has any parameters
        params = list(model.parameters())
        if params:
            self.opt = torch.optim.Adam(params, lr=base_lr, weight_decay=weight_decay)
        else:
            self.opt = None
    
    def train_step(self, users, target_ratings):
        if self.opt:
            self.opt.zero_grad()
        predicted_ratings = self.model(users)
        
        if isinstance(predicted_ratings, np.ndarray):
            predicted_ratings = torch.from_numpy(predicted_ratings).to(world.device)
        
        if predicted_ratings.device != target_ratings.device:
            target_ratings = target_ratings.to(predicted_ratings.device)
        
        loss = torch.mean((predicted_ratings - target_ratings) ** 2)
        
        if self.opt:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()
        
        return loss.cpu().item()


def create_target_ratings(dataset, users, device=None):
    """Create target ratings"""
    if device is None:
        device = world.device
        
    batch_size = len(users)
    n_items = dataset.m_items
    target_ratings = torch.zeros(batch_size, n_items, device=device)
    
    for i, user in enumerate(users):
        pos_items = dataset.allPos[user]
        if len(pos_items) > 0:
            target_ratings[i, pos_items] = 1.0
    
    return target_ratings


def train_epoch(dataset, model, loss_class, epoch, config):
    """Train for one epoch"""
    model.train()
    n_users = dataset.n_users
    train_batch_size = config['train_u_batch_size']
    
    if train_batch_size == -1:
        train_batch_size = n_users
        users_per_epoch = n_users
    else:
        users_per_epoch = min(n_users, max(1000, n_users // 4))
    
    # Sample users
    np.random.seed(epoch * 42)
    sampled_users = np.random.choice(n_users, users_per_epoch, replace=False)
    sampled_users = [int(u) for u in sampled_users]
    
    # Train in batches
    total_loss = 0.0
    n_batches = max(1, users_per_epoch // train_batch_size)
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * train_batch_size
        end_idx = min(start_idx + train_batch_size, users_per_epoch)
        user_indices = sampled_users[start_idx:end_idx]
        
        users = torch.LongTensor(user_indices).to(world.device)
        target_ratings = create_target_ratings(dataset, user_indices, device=world.device)
        
        batch_loss = loss_class.train_step(users, target_ratings)
        total_loss += batch_loss
        
        if train_batch_size == n_users:
            break
    
    return total_loss / n_batches


def test_one_batch(X):
    """Copied exactly from static model"""
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}

def evaluate(dataset, model, data_dict, config):
    """Evaluate model - copied exactly from static model"""
    if len(data_dict) == 0:
        return {'recall': np.zeros(len(world.topks)),
                'precision': np.zeros(len(world.topks)),
                'ndcg': np.zeros(len(world.topks))}
    
    u_batch_size = config['eval_u_batch_size']
    testDict = data_dict
    max_K = max(world.topks)
    
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    
    users = list(testDict.keys())
    try:
        assert u_batch_size <= len(users) / 10
    except AssertionError:
        print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
    
    users_list = []
    rating_list = []
    groundTrue_list = []
    
    # Simple batch processing for users - EXACTLY like static model
    total_batch = len(users) // u_batch_size + 1
    for i in range(0, len(users), u_batch_size):
        batch_users = users[i:i + u_batch_size]
        allPos = dataset.getUserPosItems(batch_users)
        groundTrue = [testDict[u] for u in batch_users]
        
        # Get ratings from model
        rating = model.getUsersRating(batch_users, world.dataset)
        rating = torch.from_numpy(rating) if isinstance(rating, np.ndarray) else rating
        
        # Exclude training items - EXACTLY like static model
        exclude_index = []
        exclude_items = []
        for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i] * len(items))
            exclude_items.extend(items)
        rating[exclude_index, exclude_items] = -(1<<10)
        _, rating_K = torch.topk(rating, k=max_K)
        rating = rating.cpu().numpy()
        del rating
        users_list.append(batch_users)
        rating_list.append(rating_K.cpu())
        groundTrue_list.append(groundTrue)
    
    assert total_batch == len(users_list)
    X = zip(rating_list, groundTrue_list)
    
    # Process batches
    pre_results = []
    for x in X:
        pre_results.append(test_one_batch(x))
    
    # Aggregate results - EXACTLY like static model
    for result in pre_results:
        results['recall'] += result['recall']
        results['precision'] += result['precision']
        results['ndcg'] += result['ndcg']
    results['recall'] /= float(len(users))
    results['precision'] /= float(len(users))
    results['ndcg'] /= float(len(users))
    
    return results


def compute_metrics(ground_truth, predictions):
    """Compute metrics"""
    relevance = utils.getLabel(ground_truth, predictions)
    
    recall, precision, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(ground_truth, relevance, k)
        recall.append(ret['recall'])
        precision.append(ret['precision'])
        ndcg.append(utils.NDCGatK_r(ground_truth, relevance, k))
    
    return {'recall': np.array(recall),
            'precision': np.array(precision),
            'ndcg': np.array(ndcg)}


def train_and_evaluate(dataset, model, config):
    """Complete training pipeline"""
    
    # Check validation
    has_validation = hasattr(dataset, 'valDict') and len(dataset.valDict) > 0
    
    # Initialize MSE loss
    loss_class = MSELoss(model, config)
    best_ndcg = 0.0
    best_epoch = 0
    best_model_state = None
    no_improvement = 0
    
    # Training parameters
    total_epochs = config['epochs']
    patience = config['patience']
    min_delta = config['min_delta']
    eval_every = config['n_epoch_eval']
    
    model = model.to(world.device)
    
    # Check if model has parameters
    has_params = len(list(model.parameters())) > 0
    
    if not has_params:
        print("\n📊 Model has no learnable parameters - running in static mode")
        print("Skipping training and evaluating directly...")
        
        # Use the same evaluation as static model
        print("test_u_batch_size is too big for this dataset, try a small one 94")
        results = evaluate(dataset, model, dataset.testDict, config)
        
        print(f"🎯 Test Results:")
        print(f"   Recall@20: {results['recall'][0]:.6f}")
        print(f"   Precision@20: {results['precision'][0]:.6f}")
        print(f"   NDCG@20: {results['ndcg'][0]:.6f}")
        
        return model, results
    
    # Training loop
    training_start = time()
    
    for epoch in tqdm(range(total_epochs), desc="Training"):
        # Train
        avg_loss = train_epoch(dataset, model, loss_class, epoch, config)
        
        # Evaluate
        if (epoch + 1) % eval_every == 0 or epoch == total_epochs - 1:
            eval_data = dataset.valDict if has_validation else dataset.testDict
            eval_name = "validation" if has_validation else "test"
            
            results = evaluate(dataset, model, eval_data, config)
            current_ndcg = results['ndcg'][0]
            
            # Check improvement
            if current_ndcg > best_ndcg + min_delta:
                best_ndcg = current_ndcg
                best_epoch = epoch + 1
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improvement = 0
                
                print(f"\nEpoch {epoch+1}: New best {eval_name} NDCG = {current_ndcg:.6f}")
            else:
                no_improvement += 1
                print(f"\nEpoch {epoch+1}: {eval_name} NDCG = {current_ndcg:.6f} (best: {best_ndcg:.6f})")
            
            # Early stopping
            if no_improvement >= patience // eval_every:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_model_state is not None:
        print(f"Restoring best model from epoch {best_epoch}")
        model.load_state_dict(best_model_state)
        model = model.to(world.device)
    
    training_time = time() - training_start
    
    # Final evaluation
    print("Final test evaluation...")
    final_results = evaluate(dataset, model, dataset.testDict, config)
    
    print(f"\n🎯 \033[96mFinal Results:\033[0m ⏱️ {training_time:.2f}s | 🏆 Epoch {best_epoch}")
    print(f"📊 R@20: \033[94m{final_results['recall'][0]:.4f}\033[0m | P@20: \033[94m{final_results['precision'][0]:.4f}\033[0m | NDCG@20: \033[95m{final_results['ndcg'][0]:.4f}\033[0m")
    
    return model, final_results


def BPR_train_learnable(dataset, model, optimizer, neg_ratio=1, batch_size=2048):
    """
    BPR training for learnable spectral model
    """
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
        
        # Add regularization based on filter parameters
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


def Test(dataset, model, epoch, config=None):
    """Test function compatible with static model interface"""
    if config is None:
        config = world.config
    
    return evaluate(dataset, model, dataset.testDict, config)
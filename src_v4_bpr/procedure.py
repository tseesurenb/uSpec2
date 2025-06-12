'''
Created on June 12, 2025
Improved training procedure with BPR loss and advanced negative sampling
Based on DySimGCF's successful methodology

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
import os
import pickle


class BPRLoss:
    """BPR Loss with multiple negative sampling (borrowed from DySimGCF)"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        base_lr = config['lr']
        weight_decay = config['decay']
        self.opt = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    
    def train_step(self, users, pos_items, neg_items, user_embeds, pos_embeds, neg_embeds):
        self.opt.zero_grad()
        
        # Compute BPR loss with margin (from DySimGCF)
        margin = self.config.get('margin', 0.1)
        samples = self.config.get('samples', 1)
        
        # Compute regularization loss
        if samples == 1:
            neg_reg_loss = neg_embeds.norm(2).pow(2)
        else:
            neg_reg_loss = neg_embeds.norm(2, dim=2).pow(2).sum() / neg_embeds.shape[1]
        
        reg_loss = (1 / 2) * (
            user_embeds.norm(2).pow(2) +
            pos_embeds.norm(2).pow(2) +
            neg_reg_loss
        ) / float(len(users))
        
        # Compute positive and negative scores
        pos_scores = torch.sum(user_embeds * pos_embeds, dim=1)
        
        if samples == 1:
            # Single negative case
            neg_scores = torch.sum(user_embeds * neg_embeds, dim=1)
            bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores + margin))
        else:
            # Multiple negatives case
            user_embeds_expanded = user_embeds.unsqueeze(1)
            neg_scores = torch.sum(user_embeds_expanded * neg_embeds, dim=2)
            pos_scores_expanded = pos_scores.unsqueeze(1)
            bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores_expanded + margin))
        
        # Total loss
        total_loss = bpr_loss + self.config.get('r_loss_w', 1.0) * self.config['decay'] * reg_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.opt.step()
        
        return bpr_loss.cpu().item(), reg_loss.cpu().item(), total_loss.cpu().item()


def precompute_all_epochs_samples(train_data, n_users, n_items, num_epochs, samples=1, seed=42, save_dir="../cache"):
    """Precompute negative samples for all epochs (borrowed from DySimGCF)"""
    os.makedirs(save_dir, exist_ok=True)
    
    config_signature = f"{world.dataset}_seed{seed}_epochs{num_epochs}_samples{samples}"
    filename = os.path.join(save_dir, f"spectral_precomputed_{config_signature}.pkl")
    
    # Try to load from file
    if os.path.exists(filename):
        print(f"Loading precomputed samples from {filename}")
        with open(filename, "rb") as f:
            return pickle.load(f)
    
    print(f"Precomputing samples for {num_epochs} epochs...")
    np.random.seed(seed)
    
    # Create adjacency list for fast negative sampling
    user_pos_items = {}
    all_items = set(range(n_items))
    
    for user_id in range(n_users):
        user_interactions = train_data[train_data[:, 0] == user_id, 1]
        user_pos_items[user_id] = set(user_interactions)
    
    all_epoch_data = []
    
    for epoch in tqdm(range(num_epochs), desc="Generating epoch samples"):
        # Shuffle training data for this epoch
        epoch_seed = seed + epoch
        np.random.seed(epoch_seed)
        shuffled_indices = np.random.permutation(len(train_data))
        epoch_data = train_data[shuffled_indices]
        
        users = epoch_data[:, 0]
        pos_items = epoch_data[:, 1]
        
        # Generate negative samples
        if samples == 1:
            neg_items = np.array([
                np.random.choice(list(all_items - user_pos_items[u]))
                for u in users
            ])
        else:
            neg_items = np.array([
                np.random.choice(list(all_items - user_pos_items[u]), size=samples, replace=True)
                for u in users
            ])
        
        all_epoch_data.append((users, pos_items, neg_items))
    
    # Save to file
    with open(filename, "wb") as f:
        pickle.dump(all_epoch_data, f)
    
    print(f"Saved precomputed samples to {filename}")
    return all_epoch_data


def create_target_ratings_sparse(users, pos_items, n_items, device=None):
    """Create sparse target ratings for BPR loss"""
    if device is None:
        device = world.device
    
    batch_size = len(users)
    
    # Create sparse representation
    user_indices = torch.arange(batch_size, device=device).repeat_interleave(
        torch.tensor([len(items) if isinstance(items, (list, np.ndarray)) else 1 
                     for items in pos_items], device=device)
    )
    
    if isinstance(pos_items[0], (list, np.ndarray)):
        item_indices = torch.cat([torch.tensor(items, device=device) for items in pos_items])
    else:
        item_indices = torch.tensor(pos_items, device=device)
    
    values = torch.ones(len(item_indices), device=device)
    
    sparse_ratings = torch.sparse_coo_tensor(
        torch.stack([user_indices, item_indices]), 
        values, 
        (batch_size, n_items), 
        device=device
    ).to_dense()
    
    return sparse_ratings


def train_epoch_bpr(dataset, model, loss_class, epoch_data, config):
    """Train one epoch with BPR loss"""
    model.train()
    
    users, pos_items, neg_items = epoch_data
    train_batch_size = config['train_u_batch_size']
    
    # Convert to tensors
    users = torch.tensor(users, dtype=torch.long, device=world.device)
    pos_items = torch.tensor(pos_items, dtype=torch.long, device=world.device)
    neg_items = torch.tensor(neg_items, dtype=torch.long, device=world.device)
    
    total_bpr_loss = 0.0
    total_reg_loss = 0.0
    total_loss = 0.0
    n_batches = max(1, len(users) // train_batch_size)
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * train_batch_size
        end_idx = min(start_idx + train_batch_size, len(users))
        
        batch_users = users[start_idx:end_idx]
        batch_pos = pos_items[start_idx:end_idx]
        batch_neg = neg_items[start_idx:end_idx]
        
        # Get embeddings from the model
        # For spectral model, we use the embedding layers
        user_embeds = model.user_embedding(batch_users)
        pos_embeds = model.item_embedding(batch_pos)
        
        if batch_neg.dim() == 1:
            neg_embeds = model.item_embedding(batch_neg)
        else:
            neg_embeds = model.item_embedding(batch_neg)
        
        # Train step
        bpr_loss, reg_loss, batch_total_loss = loss_class.train_step(
            batch_users, batch_pos, batch_neg, user_embeds, pos_embeds, neg_embeds
        )
        
        total_bpr_loss += bpr_loss
        total_reg_loss += reg_loss
        total_loss += batch_total_loss
    
    return {
        'bpr_loss': total_bpr_loss / n_batches,
        'reg_loss': total_reg_loss / n_batches,
        'total_loss': total_loss / n_batches
    }


def evaluate_spectral(dataset, model, data_dict, config):
    """Evaluate spectral model with proper metrics"""
    if len(data_dict) == 0:
        return {'recall': np.zeros(len(world.topks)),
                'precision': np.zeros(len(world.topks)),
                'ndcg': np.zeros(len(world.topks))}
    
    model.eval()
    eval_batch_size = config['eval_u_batch_size']
    max_K = max(world.topks)
    
    results = {'recall': np.zeros(len(world.topks)),
               'precision': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    
    with torch.no_grad():
        users = list(data_dict.keys())
        all_results = []
        
        # Create interaction tensor for masking
        interactions_tensor = torch.zeros(dataset.n_users, dataset.m_items, device=world.device)
        for user_idx in range(dataset.n_users):
            pos_items = dataset.allPos[user_idx]
            if len(pos_items) > 0:
                interactions_tensor[user_idx, pos_items] = 1.0
        
        for batch_users in utils.minibatch(users, batch_size=eval_batch_size):
            batch_users = [int(u) for u in batch_users]
            
            ground_truth = [data_dict[u] for u in batch_users]
            
            batch_users_gpu = torch.LongTensor(batch_users).to(world.device)
            ratings = model.getUsersRating(batch_users_gpu)
            
            if isinstance(ratings, np.ndarray):
                ratings = torch.from_numpy(ratings).to(world.device)
            
            # Mask out training items
            ratings = ratings * (1 - interactions_tensor[batch_users_gpu])
            
            # Get top-K predictions
            _, top_items = torch.topk(ratings, k=max_K)
            
            # Compute metrics
            batch_result = compute_metrics(ground_truth, top_items.cpu().numpy())
            all_results.append(batch_result)
        
        # Aggregate results
        for result in all_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        
        # Average
        n_users = len(users)
        results['recall'] /= n_users
        results['precision'] /= n_users
        results['ndcg'] /= n_users
    
    return results


def compute_metrics(ground_truth, predictions):
    """Compute metrics (borrowed from utils)"""
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


def prepare_training_data(dataset):
    """Prepare training data in the format expected by the BPR trainer"""
    # Convert training data to numpy array format
    train_users = dataset.trainUser
    train_items = dataset.trainItem
    
    train_data = np.column_stack((train_users, train_items))
    
    return train_data


def train_and_evaluate_spectral(dataset, model, config):
    """Complete training pipeline with BPR loss"""
    
    # Check validation
    has_validation = hasattr(dataset, 'valDict') and len(dataset.valDict) > 0
    
    # Prepare training data
    train_data = prepare_training_data(dataset)
    
    # Precompute negative samples for all epochs
    all_epoch_data = precompute_all_epochs_samples(
        train_data, 
        dataset.n_users, 
        dataset.m_items,
        config['epochs'],
        samples=config.get('samples', 1),
        seed=world.seed
    )
    
    # Initialize BPR loss
    loss_class = BPRLoss(model, config)
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
    
    # Training loop
    training_start = time()
    
    pbar = tqdm(range(total_epochs), desc="Training Spectral CF")
    
    for epoch in pbar:
        # Train
        epoch_data = all_epoch_data[epoch]
        loss_dict = train_epoch_bpr(dataset, model, loss_class, epoch_data, config)
        
        # Update progress bar
        pbar.set_postfix({
            'BPR': f"{loss_dict['bpr_loss']:.4f}",
            'Reg': f"{loss_dict['reg_loss']:.4f}",
            'Total': f"{loss_dict['total_loss']:.4f}"
        })
        
        # Evaluate
        if (epoch + 1) % eval_every == 0 or epoch == total_epochs - 1:
            eval_data = dataset.valDict if has_validation else dataset.testDict
            eval_name = "validation" if has_validation else "test"
            
            results = evaluate_spectral(dataset, model, eval_data, config)
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
    final_results = evaluate_spectral(dataset, model, dataset.testDict, config)
    
    print(f"\n🎯 \033[96mFinal Results:\033[0m ⏱️ {training_time:.2f}s | 🏆 Epoch {best_epoch}")
    print(f"📊 R@20: \033[94m{final_results['recall'][0]:.4f}\033[0m | P@20: \033[94m{final_results['precision'][0]:.4f}\033[0m | NDCG@20: \033[95m{final_results['ndcg'][0]:.4f}\033[0m")
    
    return model, final_results
'''
Created on June 3, 2025
PyTorch Implementation of uSpec: Universal Spectral Collaborative Filtering
Fixed device handling in training procedure

@author: Tseesuren Batsuuri (tseesuren.batsuuri@hdr.mq.edu.au)
'''

import world
import numpy as np
import torch
import torch.nn as nn
import utils
from time import time
from tqdm import tqdm

class MSELoss:
    def __init__(self, model, config):
        self.model = model
        base_lr = config['lr']
        weight_decay = config['decay']
        
        # Simple optimizer setup
        try:
            filter_params = list(model.get_filter_parameters())
            other_params = list(model.get_other_parameters())
            
            if len(filter_params) > 0 and len(other_params) > 0:
                # Separate optimizers for filter and other parameters
                param_groups = [
                    {'params': filter_params, 'lr': base_lr * 2.0, 'weight_decay': weight_decay * 0.1},
                    {'params': other_params, 'lr': base_lr, 'weight_decay': weight_decay}
                ]
                self.opt = torch.optim.Adam(param_groups)
            else:
                # Single optimizer
                self.opt = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        except AttributeError:
            # Fallback for models without parameter separation
            self.opt = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    
    def train_step(self, users, target_ratings):
        """Simple training step with device consistency"""
        self.opt.zero_grad()
        predicted_ratings = self.model(users)
        
        if isinstance(predicted_ratings, np.ndarray):
            predicted_ratings = torch.from_numpy(predicted_ratings).to(world.device)
        
        # FIXED: Ensure both tensors are on the same device
        if predicted_ratings.device != target_ratings.device:
            target_ratings = target_ratings.to(predicted_ratings.device)
        
        loss = torch.mean((predicted_ratings - target_ratings) ** 2)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.opt.step()
        return loss.cpu().item()

def create_target_ratings(dataset, users, device=None):
    """Create target ratings from training data on specified device"""
    if device is None:
        device = world.device
        
    batch_size = len(users)
    n_items = dataset.m_items
    
    # FIXED: Create tensor directly on the target device
    target_ratings = torch.zeros(batch_size, n_items, device=device)
    
    for i, user in enumerate(users):
        pos_items = dataset.allPos[user]
        if len(pos_items) > 0:
            target_ratings[i, pos_items] = 1.0
    
    return target_ratings

def train_epoch(dataset, model, loss_class, epoch, config):
    """Train for one epoch with proper device handling"""
    model.train()
    n_users = dataset.n_users
    train_batch_size = config['train_u_batch_size']
    
    if train_batch_size == -1:
        train_batch_size = n_users
        users_per_epoch = n_users
    else:
        users_per_epoch = min(n_users, max(1000, n_users // 4))
    
    # Sample users for this epoch
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
        
        # FIXED: Ensure users tensor is on correct device
        users = torch.LongTensor(user_indices).to(world.device)
        
        # FIXED: Create target ratings on the same device as the model
        target_ratings = create_target_ratings(dataset, user_indices, device=world.device)
        
        batch_loss = loss_class.train_step(users, target_ratings)
        total_loss += batch_loss
        
        if train_batch_size == n_users:
            break
    
    return total_loss / n_batches

def evaluate(dataset, model, data_dict, config):
    """Evaluate model on given data with proper device handling"""
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
        
        # Process in batches
        for batch_users in utils.minibatch(users, batch_size=eval_batch_size):
            batch_users = [int(u) for u in batch_users]
            
            # Get training items and ground truth
            training_items = dataset.getUserPosItems(batch_users)
            ground_truth = [data_dict[u] for u in batch_users]
            
            # FIXED: Ensure batch_users_gpu is on correct device
            batch_users_gpu = torch.LongTensor(batch_users).to(world.device)
            ratings = model.getUsersRating(batch_users_gpu)
            
            if isinstance(ratings, np.ndarray):
                ratings = torch.from_numpy(ratings)
            
            # FIXED: Ensure ratings tensor is on CPU for processing
            if ratings.device != torch.device('cpu'):
                ratings = ratings.cpu()
            
            # Exclude training items
            for i, items in enumerate(training_items):
                if len(items) > 0:
                    ratings[i, items] = -float('inf')
            
            # Get top-K predictions
            _, top_items = torch.topk(ratings, k=max_K)
            
            # Compute metrics for this batch
            batch_result = compute_metrics(ground_truth, top_items.cpu().numpy())
            all_results.append(batch_result)
        
        # Aggregate results
        for result in all_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        
        # Average over users
        n_users = len(users)
        results['recall'] /= n_users
        results['precision'] /= n_users
        results['ndcg'] /= n_users
    
    return results

def compute_metrics(ground_truth, predictions):
    """Compute recall, precision, NDCG for a batch"""
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
    """Complete training and evaluation pipeline with device consistency"""
    
    print("="*60)
    print(f"🚀 STARTING UNIVERSAL SPECTRAL CF TRAINING")
    filter_design = getattr(model, 'filter_design', 'original').upper()
    print(f"   Filter Design: {filter_design}")
    print(f"   Device: {world.device}")
    print("="*60)
    
    # Check validation availability
    has_validation = hasattr(dataset, 'valDict') and len(dataset.valDict) > 0
    if has_validation:
        print(f"✅ Using validation split ({dataset.valDataSize:,} interactions)")
    else:
        print(f"⚠️  No validation - using test data during training")
    
    # Initialize
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
    
    print(f"📊 Training: {dataset.trainDataSize:,} interactions")
    print(f"🎯 Config: {total_epochs} epochs, patience={patience}")
    
    # FIXED: Ensure model is on correct device
    model = model.to(world.device)
    
    # Training loop
    training_start = time()
    
    for epoch in tqdm(range(total_epochs), desc="Training"):
        # Train one epoch
        avg_loss = train_epoch(dataset, model, loss_class, epoch, config)
        
        # Evaluate periodically
        if (epoch + 1) % eval_every == 0 or epoch == total_epochs - 1:
            # Use validation if available, otherwise test
            eval_data = dataset.valDict if has_validation else dataset.testDict
            eval_name = "validation" if has_validation else "test"
            
            results = evaluate(dataset, model, eval_data, config)
            current_ndcg = results['ndcg'][0]
            
            # Check for improvement
            if current_ndcg > best_ndcg + min_delta:
                best_ndcg = current_ndcg
                best_epoch = epoch + 1
                # FIXED: Save model state properly handling device
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improvement = 0
                
                print(f"\n✅ Epoch {epoch+1}: New best {eval_name} NDCG = {current_ndcg:.6f}")
                print(f"   Training loss: {avg_loss:.6f}")
            else:
                no_improvement += 1
                print(f"\n📈 Epoch {epoch+1}: {eval_name} NDCG = {current_ndcg:.6f} (best: {best_ndcg:.6f})")
                print(f"   Training loss: {avg_loss:.6f}")
            
            # Early stopping
            if no_improvement >= patience // eval_every:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_model_state is not None:
        print(f"\n🔄 Restoring best model from epoch {best_epoch}")
        # FIXED: Load state dict and move to correct device
        model.load_state_dict(best_model_state)
        model = model.to(world.device)
    
    training_time = time() - training_start
    
    # Final test evaluation
    print(f"\n" + "="*60)
    print("🏆 FINAL TEST EVALUATION")
    print("="*60)
    
    final_results = evaluate(dataset, model, dataset.testDict, config)
    
    print(f"⏱️  Training time: {training_time:.2f}s")
    print(f"🎯 Best epoch: {best_epoch}")
    print(f"📊 Final test results:")
    print(f"   Recall@20:    {final_results['recall'][0]:.6f}")
    print(f"   Precision@20: {final_results['precision'][0]:.6f}")
    print(f"   NDCG@20:      {final_results['ndcg'][0]:.6f}")
    print("="*60)
    
    # Show final filter learning results
    print(f"\n--- Final Filter Learning Results ---")
    model.debug_filter_learning()
    
    return model, final_results
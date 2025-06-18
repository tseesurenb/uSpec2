"""
Main script for Raw Symmetric Softmax CF
Simple and focused - replaces degree normalization with learnable softmax
"""
import torch
import numpy as np
import time

from config import parse_args, get_config
from model import RawSymmetricSoftmax
from dataloader import Loader
import utils


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    args = parse_args()
    config = get_config(args)
    
    set_seed(config['seed'])
    
    print("=" * 60)
    print("Raw Symmetric Softmax CF")
    print("=" * 60)
    print(f"Dataset: {config['dataset']}")
    print(f"Initial temperature: {config['temperature']}")
    print(f"Device: {config['device']}")
    print("=" * 60 + "\n")
    
    # Load dataset
    dataset = Loader(config)
    
    # Create model
    model = RawSymmetricSoftmax(dataset.UserItemNet, config['temperature']).to(config['device'])
    
    # Optimizer (only temperature parameter)
    optimizer = torch.optim.Adam([model.temperature], lr=config['lr'])
    
    # Training loop
    best_ndcg = 0
    
    for epoch in range(config['epochs']):
        model.train()
        
        # Simple MSE training
        users = np.random.choice(dataset.n_users, 1000, replace=False)
        users = torch.tensor(users, dtype=torch.long).to(config['device'])
        
        # Create target
        target = torch.zeros(len(users), dataset.m_items, device=config['device'])
        for i, user in enumerate(users.cpu().numpy()):
            pos_items = dataset.allPos[user]
            if len(pos_items) > 0:
                target[i, pos_items] = 1.0
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(users)
        loss = torch.mean((pred - target) ** 2)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Evaluation
        if (epoch + 1) % config['eval_freq'] == 0:
            model.eval()
            with torch.no_grad():
                results = evaluate(dataset, model)
                ndcg = results['ndcg'][0]
                recall = results['recall'][0]
                
                print(f"Epoch {epoch+1}/{config['epochs']}")
                print(f"Loss: {loss.item():.4f} | Temperature: {model.temperature.item():.4f}")
                print(f"NDCG@20: {ndcg:.4f} | Recall@20: {recall:.4f}")
                
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
    
    print(f"\nBest NDCG@20: {best_ndcg:.4f}")


def evaluate(dataset, model):
    """Simple evaluation"""
    u_batch_size = 500
    testDict = dataset.testDict
    
    results = {'precision': np.zeros(1), 'recall': np.zeros(1), 'ndcg': np.zeros(1)}
    users = list(testDict.keys())
    
    if len(users) == 0:
        return results
    
    rating_list = []
    groundTrue_list = []
    
    for i in range(0, len(users), u_batch_size):
        batch_users = users[i:i + u_batch_size]
        allPos = dataset.getUserPosItems(batch_users)
        groundTrue = [testDict[u] for u in batch_users]
        
        rating = model.getUsersRating(batch_users)
        rating = torch.from_numpy(rating)
        
        # Exclude training items
        exclude_index = []
        exclude_items = []
        for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i] * len(items))
            exclude_items.extend(items)
        rating[exclude_index, exclude_items] = -(1<<10)
        
        _, rating_K = torch.topk(rating, k=20)
        rating_list.append(rating_K.cpu())
        groundTrue_list.append(groundTrue)
    
    # Compute metrics
    for rating, groundTrue in zip(rating_list, groundTrue_list):
        sorted_items = rating.numpy()
        r = utils.getLabel(groundTrue, sorted_items)
        
        for k in [20]:
            ret = utils.RecallPrecision_ATk(groundTrue, r, k)
            results['precision'] += ret['precision']
            results['recall'] += ret['recall']
            results['ndcg'] += utils.NDCGatK_r(groundTrue, r, k)
    
    results['recall'] /= float(len(users))
    results['precision'] /= float(len(users))
    results['ndcg'] /= float(len(users))
    
    return results


if __name__ == "__main__":
    main()
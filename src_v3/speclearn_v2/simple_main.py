"""
Even Simpler Version - No Training, Just Different Temperature Values
Like GF-CF but with symmetric softmax instead of degree normalization
"""
import torch
import numpy as np
import time

from config import parse_args, get_config
from model import RawSymmetricSoftmax
from dataloader import Loader
import utils


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


def main():
    args = parse_args()
    config = get_config(args)
    
    print("=" * 60)
    print("Raw Symmetric Softmax CF (No Training)")
    print("=" * 60)
    print(f"Dataset: {config['dataset']}")
    print("=" * 60 + "\n")
    
    # Load dataset
    dataset = Loader(config)
    
    # Test different temperature values
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    best_ndcg = 0
    best_temp = 1.0
    
    for temp in temperatures:
        print(f"Testing temperature: {temp}")
        
        # Create model with fixed temperature
        model = RawSymmetricSoftmax(dataset.UserItemNet, temp)
        model.eval()
        
        # Evaluate
        with torch.no_grad():
            results = evaluate(dataset, model)
            ndcg = results['ndcg'][0]
            recall = results['recall'][0]
            
            print(f"NDCG@20: {ndcg:.4f} | Recall@20: {recall:.4f}")
            
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_temp = temp
        
        print()
    
    print(f"Best Temperature: {best_temp}")
    print(f"Best NDCG@20: {best_ndcg:.4f}")


if __name__ == "__main__":
    main()
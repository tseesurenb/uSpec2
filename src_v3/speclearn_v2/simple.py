"""
Simple standalone script for v2
"""
import torch
import numpy as np
import time

from minimal_config import parse_args, get_config
from efficient_model import EfficientRawModel
from dataloader import Loader
import utils


def evaluate(dataset, model):
    """Evaluation with progress"""
    print("Evaluating...", end=" ", flush=True)
    start = time.time()
    
    testDict = dataset.testDict
    users = list(testDict.keys())
    
    if len(users) == 0:
        return {'ndcg': np.zeros(1), 'recall': np.zeros(1)}
    
    results = {'precision': np.zeros(1), 'recall': np.zeros(1), 'ndcg': np.zeros(1)}
    
    # Process in batches
    for i in range(0, len(users), 500):
        batch_users = users[i:i + 500]
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
        
        # Compute metrics for this batch
        sorted_items = rating_K.numpy()
        r = utils.getLabel(groundTrue, sorted_items)
        
        ret = utils.RecallPrecision_ATk(groundTrue, r, 20)
        results['precision'] += ret['precision']
        results['recall'] += ret['recall']
        results['ndcg'] += utils.NDCGatK_r(groundTrue, r, 20)
    
    # Average
    results['recall'] /= float(len(users))
    results['precision'] /= float(len(users))
    results['ndcg'] /= float(len(users))
    
    print(f"✓ ({time.time() - start:.1f}s)")
    return results


def main():
    args = parse_args()
    config = get_config(args)
    
    print("=" * 50)
    print(f"Raw Symmetric Softmax CF - Dataset: {config['dataset']}")
    print("=" * 50)
    
    # Load dataset
    print("Loading dataset...", end=" ", flush=True)
    start = time.time()
    dataset = Loader(config)
    print(f"✓ ({time.time() - start:.1f}s)")
    
    # Test temperatures
    if config['temp_range']:
        temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        best_ndcg = 0
        best_temp = 1.0
        
        for temp in temperatures:
            print(f"\nTemperature: {temp}")
            
            # Create model
            model = EfficientRawModel(dataset.UserItemNet, temp)
            
            # Evaluate
            results = evaluate(dataset, model)
            ndcg = results['ndcg'][0]
            recall = results['recall'][0]
            
            print(f"NDCG@20: {ndcg:.4f} | Recall@20: {recall:.4f}")
            
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_temp = temp
        
        print("\n" + "=" * 50)
        print(f"Best Temperature: {best_temp} | Best NDCG@20: {best_ndcg:.4f}")
        print("=" * 50)
    else:
        # Test single temperature
        temp = config['temperature']
        print(f"\nTesting single temperature: {temp}")
        
        model = EfficientRawModel(dataset.UserItemNet, temp)
        results = evaluate(dataset, model)
        ndcg = results['ndcg'][0]
        recall = results['recall'][0]
        
        print(f"NDCG@20: {ndcg:.4f} | Recall@20: {recall:.4f}")


if __name__ == "__main__":
    main()
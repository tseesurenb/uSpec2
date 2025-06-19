### ML-100k

python main.py --dataset ml-100k --filter uib --user_lr 0.0005 --item_lr 0.00005 --bipartite_lr 0.0005 --epochs 80 --filter_type spectral_basis --full_training --loss mse --u 10 --i 300 --b 50

python main.py --dataset ml-100k  --user_lr 0.0005 --item_lr 0.00005 --bipartite_lr 0.0005 --epochs 80 --filter_type spectral_basis --full_training --loss mse --u 10 --i 300 --b 50 --user_temperature -1 --filter uib


Best NDCG@20: 0.5074

### GOWALLA

python main.py --dataset gowalla --filter_type spectral_basis --loss mse --full_training --u 135 --i 300 --b 400 --filter uib

### YELP2018

python main.py --dataset yelp2018 --filter uib --user_lr 0.1 --item_lr 0.01 --bipartite_lr 0.1 --epochs 60  --full_training --loss mse --u 60 --i 300 --b 400 --filer_type spectral_basis 

Best NDCG@20: 0.0530

python main.py --dataset yelp2018  --full_training --u 135 --i 300 --b 400 --filter_type spectral_basis --filter uib --loss mse --use_two_hop --epochs 200

Best NDCG@20: 0.0584


### AMAZON-BOOK


  Recommended Learning Rates for Large Datasets:

  Gowalla:

  --user_lr 0.001 --item_lr 0.0001 --bipartite_lr 0.001

  Yelp2018:

  --user_lr 0.0005 --item_lr 0.00005 --bipartite_lr 0.0005



  Rationale:

  1. Item view needs smallest LR: Item similarities are denser and more stable, so smaller steps prevent overshooting
  2. User/Bipartite can handle slightly larger LRs: These views are typically sparser and need more aggressive updates
  3. Gowalla vs Yelp2018: Gowalla is sparser, so can handle slightly larger LRs than Yelp2018

  Why these specific ratios work:

  - Item LR = User LR / 10: Item view dominates on large datasets, so keep it stable
  - Bipartite LR = User LR: Similar sparsity patterns, can use same LR
  - Scale down for denser datasets: Yelp2018 is denser than Gowalla, so use smaller absolute values

  Try these settings with your optimal configuration:
  # Gowalla
  python main.py --dataset gowalla --filter uib --user_lr 0.001 --item_lr 0.0001 --bipartite_lr 0.001 --epochs 80 --filter_type spectral_basis --full_training --loss mse --u 135 --i 300 --b 400

  # Yelp2018  
  python main.py --dataset yelp2018 --filter uib --user_lr 0.0005 --item_lr 0.00005 --bipartite_lr 0.0005 --epochs 80 --filter_type spectral_basis --full_training --loss mse --u 135 --i 300 --b 400
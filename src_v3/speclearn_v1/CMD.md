### ML-100k

python main.py --dataset ml-100k --filter uib --user_lr 0.0005 --item_lr 0.00005 --bipartite_lr 0.0005 --epochs 80 --filter_type spectral_basis --full_training --loss mse --u 10 --i 300 --b 50

python main.py --dataset ml-100k  --user_lr 0.0005 --item_lr 0.00005 --bipartite_lr 0.0005 --epochs 80 --filter_type spectral_basis --full_training --loss mse --u 10 --i 300 --b 50 --user_temperature -1 --filter uib


Best NDCG@20: 0.5074

### GOWALLA

python main.py --dataset gowalla --filter_type spectral_basis --loss mse --full_training --u 135 --i 300 --b 400 --filter uib

### YELP2018

python main.py --dataset yelp2018 --filter uib --user_lr 0.1 --item_lr 0.01 --bipartite_lr 0.1 --epochs 60  --full_training --loss mse --u 60 --i 300 --b 400 --filer_type spectral_basis 

Best NDCG@20: 0.0530

### AMAZON-BOOK
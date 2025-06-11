 
 ## ML-100k
 python main.py --model_type user_specific --dataset ml-100k --filter uib --u_n_eigen 30 --i_n_eigen 50 --b_n_eigen 200 --loss_function bce

## Gowalla


## Yelp2018

python main.py --model_type user_specific --dataset yelp2018 --shared_base  --u_n_eigen 250 --i_n_eigen 400 --b_n_eigen 600 --filter ub --patience 10 --epochs 50 --lr 0.1
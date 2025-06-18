
### ML-100k

python main.py --dataset ml-100k --u_n_eigen 25 --i_n_eigen 200 --b_n_eign 220

### GOWALLA
python main.py --dataset gowalla --u 80 --i 500 --b 800 --uf jacobi --if spectral_basis --bf jacobi

### YELP2018
python search_independent.py --dataset yelp2018 --u 400 --i 600 --b 800

python main.py --dataset yelp2018 --u 25 --i 200 --b 220 --uf enhanced_basis --if enhanced_basis --bf original
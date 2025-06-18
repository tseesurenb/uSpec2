'''
Unified Spectral CF World Configuration - Static Version
Clean and synchronized with learnable model

@author: Tseesuren Batsuuri
'''

import os
import torch
from parse import parse_args, validate_args
import multiprocessing

# Parse and validate arguments
args = parse_args()
args = validate_args(args)

config = {}

# Basic configuration
config['test_u_batch_size'] = args.test_batch
config['dataset'] = args.dataset
config['in_mat'] = args.in_mat
config['verbose'] = args.verbose

# Eigenvalue configuration
config['u_n_eigen'] = args.u
config['i_n_eigen'] = args.i
config['b_n_eigen'] = args.b

# Filter configurations
config['user_filter_design'] = args.uf
config['item_filter_design'] = getattr(args, 'if')  # 'if' is keyword, need getattr
config['bipartite_filter_design'] = args.bf
config['user_init_filter'] = args.up
config['item_init_filter'] = args.ip
config['bipartite_init_filter'] = args.bp

# Legacy configuration (keep for compatibility)
config['multicore'] = args.multicore

# System configuration
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
config['device'] = device

CORES = multiprocessing.cpu_count() // 2
seed = args.seed
dataset = args.dataset
model_name = args.model

# Supported datasets and models
all_dataset = ['ml-100k', 'lastfm', 'gowalla', 'yelp2018', 'amazon-book']
all_models = ['spectral-cf']

if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")




TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)
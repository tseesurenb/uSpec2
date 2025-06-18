import world
import utils
from world import cprint
import numpy as np
import time
import Procedure
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

# For GF-CF, we create model with adjacency matrix directly
Recmodel = register.MODELS[world.model_name](dataset.UserItemNet)

# GF-CF training (preprocessing)
Recmodel.train()

# Test GF-CF
epoch = 0
cprint("[TEST]")
Procedure.Test(dataset, Recmodel, epoch, None, world.config['multicore'])
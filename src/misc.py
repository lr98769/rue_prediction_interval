from os.path import exists
from os import makedirs
import os
import random
import torch

import numpy as np
import tensorflow as tf

device = 'cuda:0'

def create_folder(fp):
    if not exists(fp):
        makedirs(fp)

def set_seed(seed):
    tf.config.experimental.enable_op_determinism()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
def set_seed_pytorch(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
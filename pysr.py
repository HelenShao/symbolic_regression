import numpy as np
import matplotlib.pyplot as plt
import sys, os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader
from pysr import pysr, best, get_hof

# Load the data for symbolic regression
bottleneck_2 = np.load("../Bottleneck_2/Study2/Bottleneck_2_data.npy")[:,11:13]  # two features in bottleneck
norm_input = np.load("../Bottleneck_2/Study2/Bottleneck_2_data.npy")[:,0:11]     # input data from test loader

# PYSR parameters
property_names   = ["m_vir", "v_max", "v_rms", "Halo Radius", "Scale Radius", 
                    "velocity", "Angular momentum", "Spin", "b_to_a", "c_to_a", "T_u"]
binary_operators = ["plus", "mult", "sub", "pow", "div"]
unary_operators  = ["exp", "sin", "neg", "square", "cube", "logm", "logm10", "sqrtm", 
                    "sin", "cos", "tan"]

x = norm_input        # normalized input from test_loader for model 2 (Rows=examples, columns=features)
y = bottleneck_2[:,0] # first feature of bottleneck
batch_size = 64       # for annealing and mutating

# Initiate Symbolic regression with PYSR
equations = pysr(x, y, niterations=500, binary_operators= binary_operators,
                 unary_operators= unary_operators, variable_names = property_names)


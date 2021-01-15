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
import data, architecture

################################### INPUT #####################################
# Data parameters
seed         = 4
mass_per_particle = 6.56561e+11
f_rockstar   = '../../../Halo_Data/Rockstar_z=0.0.txt'

# Training Parameters
batch_size    = 1
learning_rate = 0.0018567298164386605
weight_decay  = 1.8150136635735584e-05

# Architecture parameters
input_size    = 11
n_layers      = 5
out_features  = [82, 80, 77, 29]
bottleneck    = 2

# Best trained model
f_best_model  = 'HALOS_AE_135.pt'

# Use GPUs 
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

    
#################################### DATA & MODEL #################################
#Create datasets
train_Dataset, valid_Dataset, test_Dataset = data.create_datasets(seed, batch_size, f_rockstar)

#Create Dataloaders
test_loader  = DataLoader(dataset=test_Dataset,
                          batch_size=batch_size, shuffle=False)
# Load best model
model = architecture.AutoEncoder(input_size, bottleneck, out_features, n_layers).to(device)
if os.path.exists(f_best_model):
    model.load_state_dict(torch.load(f_best_model, map_location=torch.device('cpu')))
    
    
##################### Load the features in the bottleneck ###########################
bottleneck_2_data = np.zeros((367,24), dtype=np.float32)
# first 11 columns are the input features
# next 2 columns are the bottleneck features
# last 11 columns are the output prediction features

row = -1   #This variable keeps track of the rows in dataset
bottleneck_size = 2
for input in test_loader:
    row +=1
    # add input to first row of container
    print("Input: " + str(input))
    bottleneck_2_data[row,:11] = input
    
    # Send input through each *layer* of model (-> input becomes output)
    i = -1  # to count layers
    for layer in model:
        i += 1 
        output = layer(input)
        input = output
        
        # Add the 2 bottleneck features (from layer 9)
        if i == 8:
            print("Bottleneck: " + str(output))
            for j in range(bottleneck_size):
                bottleneck_2_data[row, 11+j] = output.detach().numpy()[:,j]
                
        if i == 18:
            print("Output: " + str(output))
            for k in range(11):
                bottleneck_2_data[row, 13+k] = output.detach().numpy()[:,k]
          
# Save the data
np.save("Bottleneck_2_data.npy", bottleneck_2_data)

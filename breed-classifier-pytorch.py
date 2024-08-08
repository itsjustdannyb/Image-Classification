import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

from sklearn.model_selection import train_test_split
print("imports ok!")

# make the data
from data_loader import data_prep
path = r"C:\Users\Daniel\Documents\dataset\dog-breed-images\dataset"
X, y = data_prep(path, (150,150))

# rescale images
X = X / 255

# split data into train,validation and test set
X_tmp, X_test, y_tmp, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size= (0.2/0.8), random_state=42)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape, sep="\n")

transform = transforms.Compose([transforms.ToTensor()])

class DogDataset(Dataset):
    def __init__(self, data, labels, transforms):
        self.data = data
        self.labels = labels
        self.transform = transforms
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]   
        if self.transform:
            sample = self.transform(sample)

        return sample, label 

# the datasets
train_dataset = DogDataset(X_train, y_train, transform)
val_dataset = DogDataset(X_val, y_val, transform)
test_dataset = DogDataset(X_test, y_test, transform)

# create a dataloader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


for i, data in enumerate(val_loader):
    images, labels = data
    print(images[i], labels[i])
    if i == 4:
        break
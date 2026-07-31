# dataset.py
# By: Maahum Khan
# Reads raw text lists from NUAA Dataset and formats them into PyTorch DataLoaders.

# Dataset Information:
# Name: Face Anti-Spoofing Dataset Paper: Face Liveness # Detection from A Single Image with Sparse Low Rank Bilinear Discriminative Model Authors: X.Tan, Y.Li, J.Liu and L.Jiang 
# Link: https://www.kaggle.com/datasets/aleksandrpikul222/nuaaaa

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class NUAADataset(Dataset):
    def __init__(self, client_txt, imposter_txt, base_dir="nuaa_data/raw", transform=None):
        def clean_path(line):
            # Clean up newlines and strip absolute Kaggle prefixes if it's there
            line = line.strip()
            if "raw/" in line:
                line = line.split("raw/")[-1]
            return os.path.join(base_dir, line)

        with open(client_txt, "r") as f:
            client_paths = [clean_path(line) for line in f.readlines()]
            
        with open(imposter_txt, "r") as f:
            imposter_paths = [clean_path(line) for line in f.readlines()]
            
        self.img_paths = client_paths + imposter_paths
        # Label 1 = Real (Client), Label 0 = Spoof (Imposter)
        self.labels = torch.cat((
            torch.ones(len(client_paths)), 
            torch.zeros(len(imposter_paths))
        )).long()
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# Get DataLoader function
def get_dataloaders(client_train, imposter_train, client_test, imposter_test, batch_size=32):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = NUAADataset(client_train, imposter_train, transform=train_transform)
    val_ds = NUAADataset(client_test, imposter_test, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader
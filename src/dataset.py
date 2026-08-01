# dataset.py
# By: Maahum Khan
# Reads raw text lists from NUAA Dataset and formats them into PyTorch DataLoaders.

# Dataset Information:
# Paper Name: Large Crowdcollected Face Anti-Spoofing Dataset
# By: Timoshenko, et al. (2019)
# Link: https://www.kaggle.com/datasets/faber24/lcc-fasd?select=LCC_FASD

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image



class LCCFASDDataset(Dataset):
    def __init__(self, split_dir, transform=None):
        """
        split_dir: Path to directory containing 'real' and 'spoof' subfolders
                   e.g., 'lcc_fasd_data/LCC_FASD/LCC_FASD_training'
        """
        self.img_paths = []
        self.labels = []
        self.transform = transform

        real_dir = os.path.join(split_dir, "real")
        spoof_dir = os.path.join(split_dir, "spoof")

        # Load Real images (Label = 1)
        if os.path.exists(real_dir):
            for fname in os.listdir(real_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.img_paths.append(os.path.join(real_dir, fname))
                    self.labels.append(1)

        # Load Spoof images (Label = 0)
        if os.path.exists(spoof_dir):
            for fname in os.listdir(spoof_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.img_paths.append(os.path.join(spoof_dir, fname))
                    self.labels.append(0)

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label



# Get DataLoader Function
def get_dataloaders(
    dataset_type="lcc_fasd",
    batch_size=32,
    # LCC-FASD args:
    lcc_dir="lcc_fasd_data/LCC_FASD",
):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))  # Cutout forces global feature learning
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(lcc_dir, "LCC_FASD_training")
    val_dir = os.path.join(lcc_dir, "LCC_FASD_development")
        
    train_ds = LCCFASDDataset(train_dir, transform=train_transform)
    val_ds = LCCFASDDataset(val_dir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader

# Get Test DataLoader Function (Uses the evaluation directory)
def get_test_loader(
    batch_size=32,
    lcc_dir="lcc_fasd_data/LCC_FASD",
):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dir = os.path.join(lcc_dir, "LCC_FASD_evaluation")
    test_ds = LCCFASDDataset(test_dir, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Expose class names explicitly for metric reports
    test_loader.dataset.classes = ["spoof", "real"]  # Label 0: spoof, Label 1: real based on your dataset logic

    return test_loader







# OLD DATASET USED. TOO SMALL, NOT USING.
"""
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
"""
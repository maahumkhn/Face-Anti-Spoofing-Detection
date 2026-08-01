# train.py
# By: Maahum Khan
# Main execution script. Trains given backbone and prints validation accuracy per epoch, including graph of train and val loss. 

import argparse
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.dataset import get_dataloaders
from src.models import get_model


# Creates graph of training and validation loss to visualize model's training ability.
def plot_loss_curves(train_losses, val_losses, graphname="loss.png", model_name="resnet18", save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2)

    plt.title(f'Training vs. Validation Loss ({model_name.upper()})', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(save_dir, graphname)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot successfully saved to: {save_path}")


# Main function
def main():
    # Argument parsing. Makes hyperparameter tuning easier.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "vgg16"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--graphname", type=str, default="loss.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {args.model} on device: {device}")

    # Data loader defined in src/dataset.py
    train_loader, val_loader = get_dataloaders(
        lcc_dir="lcc_fasd_data/LCC_FASD",
        batch_size=args.batch_size
    )

    model = get_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

    train_losses, val_losses = [], []
    start = time.time()

    # Setup saving directory for models and derive filename from graphname
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_filename = os.path.splitext(args.graphname)[0] + ".pth"
    model_save_path = os.path.join(models_dir, model_filename)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        acc = correct / total

        print(f"Epoch [{epoch+1}/{args.epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {acc*100:.2f}%")

        # Save the model weights if this epoch achieved a new best validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Saved new best model checkpoint to: {model_save_path}")

    print(f"Finished {args.model} in {(time.time() - start)/60:.2f} mins.")

    # Save graph for paper
    plot_loss_curves(train_losses, val_losses, args.graphname, args.model)

if __name__ == "__main__":
    main()
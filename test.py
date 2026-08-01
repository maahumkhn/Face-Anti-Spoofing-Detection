# test.py
# By: Maahum Khan
# Testing the best models found for each of the two backbones. Prints accuracy, recall, precision, and F1 score, and provides Confusion Matrix pic.


import os
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

# Import custom modules provided by the user
from src.dataset import get_test_loader
from src.models import get_model

def evaluate_single_model(model_name, model_path, test_loader, device='cuda'):
    print(f"\nEvaluating {model_name} using checkpoint: {model_path}")
    
    # Instantiate the architecture using your model.py definition
    model = get_model(model_name=model_name, num_classes=2)
    
    # Load the saved state dict or full checkpoint safely
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle cases where checkpoint is a dict containing state_dict vs raw state_dict vs full model
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                model = checkpoint['model']
            else:
                try:
                    model.load_state_dict(checkpoint)
                except Exception as e:
                    print(f"Warning: Could not load state dict directly ({e}). Trying full checkpoint object...")
                    model = checkpoint
        else:
            model = checkpoint
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total_loss = test_loss / len(test_loader.dataset)
    
    # Extract class names from dataset loader
    class_names = getattr(test_loader.dataset, 'classes', ["spoof", "real"])

    # Compute performance metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)

    # Print metrics
    print("=" * 50)
    print(f"Model Architecture: {model_name.upper()}")
    print(f"Test Loss:          {total_loss:.4f}")
    print(f"Overall Accuracy:   {acc * 100:.2f}%")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall:    {recall:.4f}")
    print(f"Weighted F1-Score:  {f1:.4f}")
    print("-" * 50)
    
    print("Per-Class Breakdown:")
    for idx, name in enumerate(class_names):
        print(f"  -> Class '{name}' (Label {idx}):")
        print(f"     Precision: {class_precision[idx]:.4f}")
        print(f"     Recall:    {class_recall[idx]:.4f}")
        print(f"     F1-Score:  {class_f1[idx]:.4f}")
    print("=" * 50)

    print("\nFull Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Generate and save confusion matrix figure for my final report
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name.upper()}')
    plt.tight_layout()
    plot_filename = f'confusion_matrix_{model_name}.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved confusion matrix visualization to '{plot_filename}'")

    return acc, precision, recall, f1

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running evaluation on device: {device}")

    # Defined in Dataset.py
    test_loader = get_test_loader(batch_size=64, lcc_dir="lcc_fasd_data/LCC_FASD")

    # Define model checkpoints to test
    models_to_evaluate = {
        "resnet18": "models/RN_e7_b32_l5.pth", #My best ResNet model after hyperparameter tuning
        "vgg16": "models/VGG_dp_e4_b32_lr1e-7.pth" # My best VGG after tuning
    }

    comparison_results = {}

    for model_name, path in models_to_evaluate.items():
        if os.path.exists(path):
            try:
                acc, prec, rec, f1 = evaluate_single_model(model_name, path, test_loader, device)
                comparison_results[model_name] = {
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1-Score": f1
                }
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
        else:
            print(f"Skipping {model_name}: Checkpoint file not found at '{path}'")

    # Print final side-by-side comparison
    if comparison_results:
        print("\n" + "#" * 20 + " FINAL MODEL PERFORMANCE COMPARISON " + "#" * 20)
        for m_name, metrics in comparison_results.items():
            print(f"[{m_name.upper()}]")
            print(f"  -> Accuracy:  {metrics['Accuracy'] * 100:.2f}%")
            print(f"  -> Precision: {metrics['Precision']:.4f}")
            print(f"  -> Recall:    {metrics['Recall']:.4f}")
            print(f"  -> F1-Score:  {metrics['F1-Score']:.4f}")
        print("#" * 76)

if __name__ == '__main__':
    main()
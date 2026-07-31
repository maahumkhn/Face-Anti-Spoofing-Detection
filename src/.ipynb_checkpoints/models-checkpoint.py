# models.py
# By: Maahum Khan
# Function to get the models being used: ResNet18, VGG16. May later add ResNet34, VGG19

import torch.nn as nn
from torchvision import models

def get_model(model_name="resnet18", num_classes=2):
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {model_name}")
    return model
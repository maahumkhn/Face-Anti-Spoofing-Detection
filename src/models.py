# models.py
# By: Maahum Khan
# Function to get the models being used: ResNet18, VGG16.

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, vgg16, VGG16_Weights

def get_model(model_name="resnet18", num_classes=2):
    if model_name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Freeze all layers initially
        for param in model.parameters():
            param.requires_grad = False

        # 2. Unfreeze layer4 so it can learn liveness textures
        for param in model.layer4.parameters():
            param.requires_grad = True
            
        # Replace FC head (trainable by default)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes) 
        
        return model

    elif model_name == "vgg16":
        model = vgg16(weights=VGG16_Weights.DEFAULT)
        
        # 1. Freeze the early and middle convolutional feature blocks (up to block 5)
    for param in model.features.parameters():
        param.requires_grad = False
        
    # 2. UNFREEZE the final convolutional block (features[24:] corresponds to the last block of VGG-16)
    for param in model.features[24:].parameters():
        param.requires_grad = True
        
    # 3. Freeze the early dense layers of the classifier to prevent overfitting
    for param in model.classifier.parameters():
        param.requires_grad = False
        
    # 4. Unfreeze the last few classifier layers and replace the final output layer
    for param in model.classifier[3:].parameters():
        param.requires_grad = True
            
        # Replace ONLY the very last layer of the classifier (trainable by default)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
            #nn.ReLU(True))
            #nn.Dropout(p=0.5))
        
        return model

    else:
        raise ValueError(f"Unsupported model: {model_name}")
# Created By LORD 
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from torchsummary import summary
def resnet_model(num_classes=2):
    weights = ResNet50_Weights.DEFAULT  
    model = models.resnet50(weights=weights)


    for param in model.parameters():
        param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    return model


def densenet_model(num_classes=4):
    weights = DenseNet121_Weights.DEFAULT
    model = densenet121(weights=weights)

    model.classifier = nn.Linear(model.classifier.in_features, num_classes)


    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "denseblock4" in name or "norm5" in name or "classifier" in name:
            param.requires_grad = True

    return model

def densenet_model_v3_4(num_classes=4):
    weights = DenseNet121_Weights.DEFAULT
    model = densenet121(weights=weights)

    model.classifier = nn.Linear(model.classifier.in_features, num_classes)


    for param in model.parameters():
        param.requires_grad = False


    for name, param in model.named_parameters():
        if "denseblock4" in name or "norm5" in name or "classifier" in name:
            param.requires_grad = True
    for name, param in model.named_parameters():
        if "denseblock3" in name or "norm5" in name or "classifier" in name:
            param.requires_grad = True

    return model



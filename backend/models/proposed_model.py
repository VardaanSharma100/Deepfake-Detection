import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_l as proposed_model, EfficientNet_V2_L_Weights as proposed_model_weights

class Proposed_model(nn.Module):
    def __init__(self, num_classes=2, freeze_features=True):
        super(Proposed_model, self).__init__()
        weights = proposed_model_weights.DEFAULT
        self.backbone = proposed_model(weights=weights)
        self.backbone.avgpool = nn.AdaptiveAvgPool2d(1)
        if freeze_features:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.extra_cnn = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),  
        nn.BatchNorm2d(3),
        nn.ReLU(inplace=True)
        )
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 1280),  
            nn.SiLU(inplace=True),   
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes)
        )
    def forward(self, x):
        x=self.extra_cnn(x)
        return self.backbone(x)
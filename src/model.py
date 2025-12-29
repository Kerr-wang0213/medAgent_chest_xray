import torch.nn as nn
from torchvision import models

class ChestXRayResNet18(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=False):
        super(ChestXRayResNet18, self).__init__()
        
        print(f"[MODEL] Initialize ResNet-18 (freeze_backbone={freeze_backbone})")
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
      
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

import torch
import torch.nn as nn
from torchvision import models

class ChestXRayResNet18(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=False):
        super(ChestXRayResNet18, self).__init__()
        
        # 1. 加载权重
        print(f"[Model] Initializing ResNet-18 (freeze_backbone={freeze_backbone})...")
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 2. 冻结逻辑 (保留这个功能方便写报告凑字数/做实验)
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # 3. 修改分类头 (吸取你的优点，用 Sequential，预留扩展空间)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5), # 既然用了 Sequential，不如顺手加个 Dropout 防止过拟合？(可选)
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

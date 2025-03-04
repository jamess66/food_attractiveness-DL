import torch
import torch.nn as nn
import torchvision.models as models

class FoodComparisonModel(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True,
                 freeze_backbone=False, hidden_size=512, dropout_prob=0.3):
        super().__init__()
        
        # Load backbone
        if backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            self.feature_dim = 2048
        elif backbone_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained).features
            self.feature_dim = 1536
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Comparison classifier
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.feature_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size//2, 2)
        )

    def forward_features(self, x):
        features = self.backbone(x)
        return features.flatten(1)

    def forward(self, x1, x2):
        features1 = self.forward_features(x1)
        features2 = self.forward_features(x2)
        combined = torch.cat([features1, features2], dim=1)
        return self.classifier(combined)
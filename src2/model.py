import torch
import torch.nn as nn
import torchvision.models as models

class FoodComparisonModel(nn.Module):
    def __init__(self, backbone_name='convnext', pretrained=True,
                 freeze_backbone=False, hidden_size=512, dropout_prob=0.3):
        super().__init__()
        
        if backbone_name == 'convnext':
            weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            convnext = models.convnext_base(weights=weights)
            self.backbone = nn.Sequential(
                convnext.features,
                convnext.avgpool,
                nn.Flatten(1)
            )
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy_input)
                self.feature_dim = features.shape[1]
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
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
        return self.backbone(x)

    def forward(self, x1, x2):
        features1 = self.forward_features(x1)
        features2 = self.forward_features(x2)
        combined = torch.cat([features1, features2], dim=1)
        return self.classifier(combined)
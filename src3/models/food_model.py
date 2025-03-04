import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import SqueezeExcitation

class FoodComparisonModel(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True, 
                 emb_dim=512, dropout=0.3, num_attention_heads=4):
        super().__init__()
        
        # Base feature extractor
        self.backbone = self._build_backbone(backbone_name, pretrained)
        self.feature_dim = self._get_feature_dim()
        
        # Attention modules
        self.attention = nn.Sequential(
            SqueezeExcitation(self.feature_dim, reduction=16),
            nn.Conv2d(self.feature_dim, self.feature_dim, 3, padding=1),
            nn.GroupNorm(8, self.feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Comparison head
        self.comparison = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.MultiheadAttention(emb_dim, num_attention_heads),
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 2)
        )
        
        self._init_weights()

    def _build_backbone(self, name, pretrained):
        if name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            return nn.Sequential(*list(model.children())[:-2])
        elif name == 'efficientnet':
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b3(weights=weights)
            return model.features
        else:
            raise ValueError(f"Unsupported backbone: {name}")

    def _get_feature_dim(self):
        with torch.no_grad():
            dummy = torch.rand(1, 3, 224, 224)
            features = self.backbone(dummy)
            return features.shape[1] * features.shape[2] * features.shape[3]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.backbone(x)
        x = self.attention(x)
        return x.flatten(1)

    def forward(self, x1, x2):
        f1 = self.projection(self.forward_features(x1))
        f2 = self.projection(self.forward_features(x2))
        combined = torch.cat([f1, f2], dim=1)
        attn_output, _ = self.comparison[1](combined.unsqueeze(0), combined.unsqueeze(0), combined.unsqueeze(0))
        return self.comparison[3](self.comparison[2](attn_output.squeeze(0)))
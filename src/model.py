import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    """
    Basic ResNet block with optional downsampling
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Block, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsampling layer (if needed)
        self.downsample = downsample
        
        # Activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Store the input for residual connection
        identity = x

        # First conv layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv layer
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsampling if necessary
        if self.downsample is not None:
            identity = self.downsample(x)

        # Residual connection
        out += identity
        out = self.relu(out)

        return out

class ResNetCustom(nn.Module):
    def __init__(self, block=Block, layers=[2,2,2,2], num_classes=2):
        super(ResNetCustom, self).__init__()
        
        # Initial convolution
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Weight initialization
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Create a layer with multiple blocks
        """
        # Downsample layer for changing dimensions
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        # First block might need different handling
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        
        # Update input channels
        self.in_channels = out_channels * block.expansion
        
        # Add subsequent blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global Average Pooling
        x = self.avgpool(x)
        
        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

    def _initialize_weights(self):
        """
        Custom weight initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.food_model = ResNetCustom()
    
    def forward(self, image1, image2):
        # Get attractiveness scores for both images
        output1 = self.food_model(image1)
        output2 = self.food_model(image2)
        
        return output1, output2

def custom_loss(output1, output2, labels):
    """
    Custom loss function to compare attractiveness
    Labels: 0 means second image is more attractive
            1 means first image is more attractive
    """
    # Compute difference in attractiveness scores
    diff = torch.abs(output1[:, 1] - output2[:, 1])
    
    # Compute ranking loss
    ranking_loss = torch.mean(
        torch.where(
            labels == 1, 
            torch.relu(0.5 - diff),   # Penalize if first image is not clearly more attractive
            torch.relu(diff - 0.5)    # Penalize if second image is not clearly more attractive
        )
    )
    
    return ranking_loss

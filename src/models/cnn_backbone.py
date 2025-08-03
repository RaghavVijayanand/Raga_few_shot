import torch
import torch.nn as nn
import torchvision.models as models

class CNNBackbone(nn.Module):
    def __init__(self, out_dim=128, pretrained=True, input_height=128, input_width=216):
        super().__init__()
        # Use a small ResNet (e.g., ResNet18) as backbone
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer for 1-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the original FC layer and avgpool
        self.resnet.fc = nn.Identity()
        self.resnet.avgpool = nn.Identity()
        
        # Add our own adaptive pooling to ensure fixed output size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_proj = nn.Linear(512, out_dim)

    def forward(self, x):
        # x: (batch, 1, height, width)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.out_proj(x)
        return x

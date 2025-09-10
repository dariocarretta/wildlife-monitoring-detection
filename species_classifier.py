import torch
import torch.nn as nn
from torchvision.ops import SqueezeExcitation

# classifier model
class SpeciesClassifier(nn.Module):
    """Classifier model, implementing residual squeeze-and-excitation blocks
            to implement an "attention mechanism" to the features of the conv layers.

    Args:
        initial_features (int, optional): number of initial features of the first conv layer.
                                          Defaults to 64.
        num_classes (int, optional): number of output classes. Defaults to 27.
        dropout_rate (float, optional): dropout rate of dropout layer. Defaults to 0.2.
    """
    def __init__(self, initial_features=64, num_classes=27, dropout_rate=0.2):
        
        super().__init__()
        
        # initial convolution
        self.initial_conv = nn.Conv2d(3, initial_features, kernel_size=7, stride=2, padding="same")
        self.initial_bn = nn.BatchNorm2d(initial_features)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResSE blocks with progressive feature scaling
        self.block1 = ResSEBlock(initial_features, initial_features)  # 64 -> 64
        self.block2 = ResSEBlock(initial_features, initial_features * 2)  # 64 -> 128
        self.downsample1 = nn.MaxPool2d(kernel_size=2, stride=2)  # reduce spatial size
        
        self.block3 = ResSEBlock(initial_features * 2, initial_features * 4)  # 128 -> 256
        self.downsample2 = nn.MaxPool2d(kernel_size=2, stride=2)  # reduce spatial size
        
        # global average pooling to get 1 value per channel
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # classification head with single dropout layer for better generalization
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(initial_features * 4, initial_features * 2)  # 256 -> 128
        self.fc2 = nn.Linear(initial_features * 2, num_classes)  # 128 -> 27
        
    def forward(self, x):
        
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 224x224 -> 56x56
        
        # ResSE blocks
        x = self.block1(x)  # 56x56x64
        x = self.block2(x)  # 56x56x128
        x = self.downsample1(x)  # 28x28x128
        
        x = self.block3(x)  # 28x28x256
        x = self.downsample2(x)  # 14x14x256
        
        # global pooling and classification
        x = self.global_avg_pool(x)  # 1x1x256
        x = torch.flatten(x, 1)  # 256
        
        x = self.dropout(x)
        x = self.fc1(x)  # 128
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # 27
        
        return x
        
# TODO: add predict function to classifier

# squeeze-and-excitation block with residual connection
class ResSEBlock(nn.Module):
    def __init__(self, input_channels, output_channels, squeeze_r=16):
        """Squeeze and excitation block with two successive convolutions and a residual connection.

        Args:
            input_channels (int): number of input channels to the first conv layer.
            output_channels (int): number of output feature channels after the block.
            squeeze_r (int, optional): number of squeeze channels of the SE block. Defaults to 16.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 5, padding="same")
        self.norm1 = nn.BatchNorm2d(output_channels)
        self.SEblock1 = SqueezeExcitation(output_channels, squeeze_channels=squeeze_r)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding="same")
        self.norm2 = nn.BatchNorm2d(output_channels)
        self.SEblock2 = SqueezeExcitation(output_channels, squeeze_channels=squeeze_r)
        
        # Handle difference between input and output shapes for residual connection
        self.downsample = None
        if input_channels != output_channels:
            self.downsample = nn.Conv2d(input_channels, output_channels, 1, padding=0)

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.SEblock1(out)
        out = self.leakyrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.SEblock2(out)
        out = self.leakyrelu(out)
        
        # Residual connection
        out = out + identity
        
        return out
import torch.nn as nn
from torch import Tensor
from torchvision.ops import SqueezeExcitation


class ResSEBlock(nn.Module):
    """
    Squeeze and excitation block with two successive convolutions and a residual connection.
    The Squeeze and Excitation parts implement an attention mechanism on the result of the convolutions,
    by weighting the channels of the extracted feature blocks.
    """

    def __init__(self, input_channels: int, output_channels: int, squeeze_r: int = 16):
        """
        Args:
            input_channels (int): Number of input channels to the first conv layer.
            output_channels (int): Number of output feature channels after the block.
            squeeze_r (int, optional): Number of squeeze channels of the SE block. Defaults to 16.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 5, padding="same")
        self.norm1 = nn.BatchNorm2d(output_channels)
        self.SEblock1 = SqueezeExcitation(output_channels, squeeze_channels=squeeze_r)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding="same")
        self.norm2 = nn.BatchNorm2d(output_channels)
        self.SEblock2 = SqueezeExcitation(output_channels, squeeze_channels=squeeze_r)

        # handle difference between input and output shapes for residual connection
        self.downsample = None
        if input_channels != output_channels:
            self.downsample = nn.Conv2d(input_channels, output_channels, 1, padding=0)

    def forward(self, x: Tensor):
        identity = self.downsample(x) if self.downsample else x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.SEblock1(out)
        out = self.leakyrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.SEblock2(out)
        out = self.leakyrelu(out)

        # residual connection
        out = out + identity

        return out


class FeatureExtractorBackbone(nn.Module):
    """Backbone feature extractor for YOLO-like detector."""

    def __init__(self, init_features=32):
        """
        Args:
            init_features (int, optional): Number of convolutional units for the first ResSeBlock. Defaults to 32.
        """
        super().__init__()
        self.layers = nn.Sequential(
            # initial feature extracting layer
            nn.Conv2d(3, init_features, 3),
            nn.BatchNorm2d(init_features),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            # initial ResSEBlock
            ResSEBlock(init_features, init_features),
            nn.MaxPool2d(2, 2),
            # deeper feature learning
            ResSEBlock(init_features, init_features * 2),
            ResSEBlock(init_features * 2, init_features * 2),
            nn.MaxPool2d(2, 2),
            # high-level features
            ResSEBlock(init_features * 2, init_features * 4),
            ResSEBlock(init_features * 4, init_features * 4),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x: Tensor):
        return self.layers(x)


class CustomClassifier(nn.Module):
    """Custom classifier with a backbone feature extractor based on SqueezeExcitation blocks and residual connections."""

    def __init__(
        self, num_classes: int = 3, init_features: int = 32, dropout_rate: float = 0.3
    ):
        """
        Args:
            num_classes (int): Number of species classes to classify. Defaults to 3.
            init_features (int, optional): Number of initial convolutional features. Defaults to 32.
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.3.
        """
        super().__init__()
        # backbone feature extractor
        self.backbone = FeatureExtractorBackbone(init_features=init_features)
        final_features = init_features * 4  # output from backbone
        # global average pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(final_features, final_features // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(final_features // 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)  # (batch, channels, H, W)
        pooled = self.global_pool(features)  # (batch, channels, 1, 1)
        pooled = pooled.flatten(1)  # (batch, channels)
        logits = self.classifier(pooled)

        return logits


if __name__ == "__main__":
    det = CustomClassifier(init_features=32)

    print(det, "\n")
    print("-" * 40, end="\n\n")
    print(
        f"Number of learnable parameters: {sum(p.numel() for p in det.parameters() if p.requires_grad)}"
    )

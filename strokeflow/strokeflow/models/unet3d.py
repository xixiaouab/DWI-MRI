import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock3D(nn.Module):
    """3D version of the basic ResNet block."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet34Encoder3D(nn.Module):
    """3D ResNet-34 style encoder."""
    def __init__(self, in_channels=1, base_filters=64):
        super().__init__()
        self.in_planes = base_filters

        # Initial convolution
        # Input: (B, C, D, H, W) -> Output: (B, 64, D/2, H/2, W/2)
        self.conv1 = nn.Conv3d(in_channels, base_filters, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        
        # MaxPool: (B, 64, D/2, H/2, W/2) -> (B, 64, D/4, H/4, W/4)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # ResNet-34 layers: [3, 4, 6, 3]
        # Layer 1: (B, 64, D/4, H/4, W/4)
        self.layer1 = self._make_layer(base_filters, 3, stride=1)
        # Layer 2: (B, 128, D/8, H/8, W/8)
        self.layer2 = self._make_layer(base_filters * 2, 4, stride=2)
        # Layer 3: (B, 256, D/16, H/16, W/16)
        self.layer3 = self._make_layer(base_filters * 4, 6, stride=2)
        # Layer 4: (B, 512, D/32, H/32, W/32)
        self.layer4 = self._make_layer(base_filters * 8, 3, stride=2)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * BasicBlock3D.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * BasicBlock3D.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * BasicBlock3D.expansion),
            )

        layers = []
        layers.append(BasicBlock3D(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * BasicBlock3D.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, D, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = x  # Skip connection 1 (Pre-Maxpool): (B, 64, D/2, H/2, W/2)

        x = self.maxpool(x)
        x = self.layer1(x)
        c2 = x  # Skip connection 2: (B, 64, D/4, H/4, W/4)

        x = self.layer2(x)
        c3 = x  # Skip connection 3: (B, 128, D/8, H/8, W/8)

        x = self.layer3(x)
        c4 = x  # Skip connection 4: (B, 256, D/16, H/16, W/16)

        x = self.layer4(x)
        c5 = x  # Bottleneck/Bridge: (B, 512, D/32, H/32, W/32)

        return c1, c2, c3, c4, c5


class UpBlock3D(nn.Module):
    """
    Standard U-Net decoder block: Upsample -> Concat -> Conv3D -> BN -> ReLU
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # Upsampling via Transposed Convolution
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        
        # Convolution after concatenation
        # Input channels = out_ch (from upsample) + skip_ch (from encoder)
        self.conv1 = nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle potential shape mismatch due to odd input dimensions in encoder
        if x.shape[2:] != skip.shape[2:]:
            diffZ = skip.size(2) - x.size(2)
            diffY = skip.size(3) - x.size(3)
            diffX = skip.size(4) - x.size(4)
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2,
                          diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([skip, x], dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class StrokeFlowUNet3D(nn.Module):
    """
    3D ResNet-34 U-Net backbone [cite: 466] with Dual Heads.
    Structure based on Table 7 in the paper.
    """
    def __init__(self, in_channels=1, base_filters=64):
        super().__init__()
        
        # Shared Encoder [cite: 466, 477]
        self.encoder = ResNet34Encoder3D(in_channels=in_channels, base_filters=base_filters)

        # Decoder Path (Symmetric U-Net)
        # Encoder channels: c1=64, c2=64, c3=128, c4=256, c5=512
        
        # Dec 1: Input c5(512), Skip c4(256) -> Output 256
        self.dec1 = UpBlock3D(in_ch=base_filters*8, skip_ch=base_filters*4, out_ch=base_filters*4)
        
        # Dec 2: Input Dec1(256), Skip c3(128) -> Output 128
        self.dec2 = UpBlock3D(in_ch=base_filters*4, skip_ch=base_filters*2, out_ch=base_filters*2)
        
        # Dec 3: Input Dec2(128), Skip c2(64) -> Output 64
        self.dec3 = UpBlock3D(in_ch=base_filters*2, skip_ch=base_filters, out_ch=base_filters)
        
        # Dec 4: Input Dec3(64), Skip c1(64) -> Output 64
        self.dec4 = UpBlock3D(in_ch=base_filters, skip_ch=base_filters, out_ch=base_filters)

        # Final Upsample: Restore to original resolution (D/2 -> D)
        self.final_up = nn.ConvTranspose3d(base_filters, base_filters, kernel_size=2, stride=2)

        # Prediction Heads [cite: 470]
        # 1. Density Head: (B, 1, D, H, W) -> Sigmoid
        self.density_head = nn.Sequential(
            nn.Conv3d(base_filters, 1, kernel_size=1),
            nn.Sigmoid() # Ensure output is [0, 1] [cite: 472]
        )
        
        # 2. Flow Head: (B, 3, D, H, W) -> Linear
        self.flow_head = nn.Conv3d(base_filters, 3, kernel_size=1) # [cite: 473]
        
        self._init_weights()

    def _init_weights(self):
        """Kaiming He initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (B, C, D, H, W)
        
        # Encoder
        c1, c2, c3, c4, c5 = self.encoder(x)
        
        # Decoder
        d1 = self.dec1(c5, c4) # 512 + 256 -> 256
        d2 = self.dec2(d1, c3) # 256 + 128 -> 128
        d3 = self.dec3(d2, c2) # 128 + 64 -> 64
        d4 = self.dec4(d3, c1) # 64 + 64 -> 64
        
        out = self.final_up(d4) # 64 -> 64, full resolution
        
        # Heads
        phi = self.density_head(out) # Scalar Density Field
        flow = self.flow_head(out)   # Vector Flow Field

        return phi, flow
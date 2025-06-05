import torch
import torch.nn as nn
import torch.nn.functional as F
from model.util.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.inner_channels = [768, 384, 192, 96]
        self.chanel_1x1conv = nn.Conv2d(self.inner_channels[0], self.inner_channels[1], 1)
        self.fuse_3x3convs = nn.ModuleList([
            nn.Conv2d(self.inner_channels[1], self.inner_channels[2], 3, padding=1),
            nn.Conv2d(self.inner_channels[2], self.inner_channels[3], 3, padding=1),
            nn.Conv2d(self.inner_channels[3], self.inner_channels[3]//2, 3, padding=1),
            nn.Conv2d(self.inner_channels[3]//2, self.inner_channels[3]//4, 3, padding=1),
        ])
        self.bn_after3x3convs = nn.ModuleList([
            nn.BatchNorm2d(self.inner_channels[2]),
            nn.BatchNorm2d(self.inner_channels[3]),
            nn.BatchNorm2d(self.inner_channels[3]//2),
            nn.BatchNorm2d(self.inner_channels[3]//4),
        ])
        self.gelu = nn.GELU()
        self.head = nn.Sequential(nn.Conv2d(self.inner_channels[3]//4, num_classes, kernel_size=3, padding=1),
                                  nn.Sigmoid())

        self._init_weight()

    def up_add(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='bilinear', align_corners=True)
        if lateral is not None:
            return lateral + top2x
        return top2x

    def forward(self, x, low_level_feat):
        x = self.chanel_1x1conv(x)
        # 16
        x = self.up_add(x, low_level_feat[-1])
        x = self.fuse_3x3convs[0](x)
        x = self.bn_after3x3convs[0](x)
        x = self.gelu(x)
        # 32
        x = self.up_add(x, low_level_feat[-2])
        x = self.fuse_3x3convs[1](x)
        x = self.bn_after3x3convs[1](x)
        x = self.gelu(x)
        # 64
        x = self.up_add(x, low_level_feat[-3])
        x = self.fuse_3x3convs[2](x)
        x = self.bn_after3x3convs[2](x)
        x = self.gelu(x)
        # 128
        x = self.up_add(x, None)
        x = self.fuse_3x3convs[3](x)
        x = self.bn_after3x3convs[3](x)
        x = self.gelu(x)
        # 256
        x = self.up_add(x, None)

        return self.head(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes):
    return Decoder(num_classes)


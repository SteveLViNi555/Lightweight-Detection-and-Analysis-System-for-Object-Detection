import torch
import torch.nn as nn
from torchvision import models
from .utils import generate_priors

class SSD(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=21, image_size=300):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        in_channels_list = [128, 256, 512]
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        # 先初始化 priors，获取每个特征图的 anchor 数
        self._initialize_priors()
        # 根据实际 anchor 数创建卷积层
        for in_ch, num_a in zip(in_channels_list, self.anchors_per_fm):
            self.loc_layers.append(nn.Conv2d(in_ch, num_a * 4, kernel_size=3, padding=1))
            self.conf_layers.append(nn.Conv2d(in_ch, num_a * self.num_classes, kernel_size=3, padding=1))

    def _initialize_priors(self):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            x = self.backbone.conv1(dummy)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            h2 = x.shape[2]
            x = self.backbone.layer3(x)
            h3 = x.shape[2]
            x = self.backbone.layer4(x)
            h4 = x.shape[2]
            self.feature_sizes = [h2, h3, h4]
            print(f"实际特征图尺寸: {self.feature_sizes}")

            strides = [self.image_size // f for f in self.feature_sizes]
            # anchor 尺寸匹配特征图步长: stride 4→30px, stride 8→90px, stride 16→210px
            min_sizes = [30, 90, 180]
            max_sizes = [90, 180, 300]
            aspect_ratios = [[2, 3], [2, 3], [2, 3]]
            self.priors, self.anchors_per_fm = generate_priors(
                self.feature_sizes, self.image_size,
                strides, min_sizes, max_sizes, aspect_ratios)
            expected = sum(f * f * a for f, a in zip(self.feature_sizes, self.anchors_per_fm))
            assert self.priors.shape[0] == expected, \
                f"Priors count {self.priors.shape[0]} != expected {expected}"
            print(f"生成的锚框总数: {self.priors.shape[0]}")

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        source2 = x
        x = self.backbone.layer3(x)
        source3 = x
        x = self.backbone.layer4(x)
        source4 = x
        sources = [source2, source3, source4]

        loc_outputs, conf_outputs = [], []
        for feat, loc_conv, conf_conv in zip(sources, self.loc_layers, self.conf_layers):
            loc = loc_conv(feat)
            conf = conf_conv(feat)
            batch, _, H, W = loc.shape
            loc = loc.permute(0, 2, 3, 1).contiguous().view(batch, -1, 4)
            conf = conf.permute(0, 2, 3, 1).contiguous().view(batch, -1, self.num_classes)
            loc_outputs.append(loc)
            conf_outputs.append(conf)

        loc = torch.cat(loc_outputs, 1)
        conf = torch.cat(conf_outputs, 1)
        return loc, conf
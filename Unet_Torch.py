import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Сверточные слои вниз
        self.down_conv1 = self.double_conv(1, 32)
        self.down_conv2 = self.double_conv(32, 64)

        # Подвыборка
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Сверточные слои вверх
        self.up_conv1 = self.double_conv(64, 64)
        self.up_conv2 = self.double_conv(128, 32)

        # Повышающая дискретизация
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Финальные сверточные слои
        self.final_conv1 = self.double_conv(64, 32)
        self.final_conv2 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Сверточные слои вниз
        conv1 = self.down_conv1(x)
        x = self.max_pool(conv1)
        conv2 = self.down_conv2(x)
        x = self.max_pool(conv2)

        # Сверточные слои вверх
        x = self.up_sample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.up_conv1(x)
        x = self.up_sample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.up_conv2(x)

        # Финальные сверточные слои
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        x = self.sigmoid(x)

        return x
import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(384, 384, 3, 1, groups=2),
        #     nn.BatchNorm2d(384),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv5 = nn.Conv2d(384, 256, 3, 1, groups=2)

    def forward(self, x):
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        # x_conv4 = self.conv4(x_conv3)
        # x_conv5 = self.conv5(x_conv4)
        return x_conv3
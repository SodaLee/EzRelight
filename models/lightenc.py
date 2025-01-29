# a simple mlp encoder for light field images
# input: light field images 3 x 32 x 32
# processing: 3 mlp layers with 4096, 4096, 4096 neurons respectively
# output: latent vector 2304

import torch
import torch.nn as nn
import torch.nn.functional as F

class LightEnc(nn.Module):
    def __init__(self):
        super(LightEnc, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 2048*3)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
# a 5-layer MLP with hidden state 128, and input/output same as latent channels for different models
class MLP5(nn.Module):
    def __init__(self, hidden_size=128):
        super(MLP5, self).__init__()
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=self.hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=self.hidden_size, out_channels=4, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x

class DepthFusion(nn.Module):
    def __init__(self, hidden_size=128):
        super(DepthFusion, self).__init__()
        # 编码器
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 空洞卷积层
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=4, dilation=4),
            nn.ReLU()
        )
        # 解码器
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # 输出层
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # 编码器
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        # 空洞卷积
        d1 = self.dilated_conv(p2)
        # 解码器
        u1 = self.up1(d1)
        c1 = torch.cat([u1, e2], dim=1)
        d2 = self.decoder1(c1)
        u2 = self.up2(d2)
        c2 = torch.cat([u2, e1], dim=1)
        d3 = self.decoder2(c2)
        # 输出
        out = self.final_conv(d3)
        return out
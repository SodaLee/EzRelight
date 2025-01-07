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
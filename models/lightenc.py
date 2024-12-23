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
        self.fc4 = nn.Linear(4096, 2304)

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
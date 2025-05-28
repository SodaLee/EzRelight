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
    def __init__(self, in_channels=2, feature_dim=64):
        """
        Args:
            in_channels: 输入通道数（fg_depth 和 bg_depth）
            feature_dim: 中间特征通道
        """
        super(DepthFusion, self).__init__()

        # 仿射变换预测（全图级别）
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.affine_regressor = nn.Linear(feature_dim, 2)

        # 动态底部区域深度偏移估计
        self.depth_bias_predictor = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 1)
        )

    def forward(self, depth, fg_mask):
        """
        Args:
            depth: Tensor [B, 2, H, W] → 前景深度 + 背景深度
            fg_mask: Tensor [B, 1, H, W], range ∈ [0, 1]（显式前景掩码）
        Returns:
            fg_depth_aligned: [B, 1, H, W] — 对齐后的前景深度图
        """
        B, C, H, W = depth.shape
        assert C == 2 and fg_mask.shape == (B, 1, H, W)

        fg_depth = depth[:, 0:1, :, :]
        bg_depth = depth[:, 1:2, :, :]

        # ---------- Step 1: 全局仿射变换 ----------
        global_feat = self.encoder(depth).view(B, -1)
        affine_params = self.affine_regressor(global_feat)  # [B, 2]
        scale = affine_params[:, 0].view(B, 1, 1, 1)
        shift = affine_params[:, 1].view(B, 1, 1, 1)
        fg_depth_affine = fg_depth * scale + shift

        # ---------- Step 2: 动态底部区域提取 ----------
        # 对每个样本独立处理
        depth_bias_list = []
        for b in range(B):
            mask = fg_mask[b, 0]  # [H, W]
            mask_sum = mask.sum()

            if mask_sum < 1e-4:
                # 极端情况：前景缺失
                depth_bias_list.append(torch.tensor([0.0], device=depth.device))
                continue

            # 获取前景区域中每个像素的 y 坐标加权平均 → 得到重心 y 坐标
            y_coords = torch.arange(H, device=depth.device).float().view(H, 1).expand(H, W)
            fg_y_center = (mask * y_coords).sum() / (mask_sum + 1e-6)

            # 选取重心以下（或某范围）的像素作为“底部前景区域”
            delta = int(0.2 * H)  # 可调：底部区域高度（如20%）
            y_start = int(min(H - 1, max(0, fg_y_center.item())))
            y_end = min(H, y_start + delta)

            # 构造底部 mask（在前景 mask 内 + 位于底部）
            bottom_mask = torch.zeros_like(mask)
            bottom_mask[y_start:y_end, :] = 1.0
            bottom_mask = bottom_mask * mask  # 同时满足两个条件

            # 应用到底部区域的深度
            fg_bot = fg_depth_affine[b, 0] * bottom_mask
            bg_bot = bg_depth[b, 0] * bottom_mask
            bottom_pair = torch.stack([fg_bot, bg_bot], dim=0).unsqueeze(0)  # [1, 2, H, W]

            bias = self.depth_bias_predictor(bottom_pair).view(1)  # [1]
            depth_bias_list.append(bias)

        # 拼接所有样本偏移值
        depth_bias = torch.stack(depth_bias_list, dim=0).view(B, 1, 1, 1)

        # ---------- Step 3: 校正输出 ----------
        fg_depth_aligned = fg_depth_affine + depth_bias
        return fg_depth_aligned
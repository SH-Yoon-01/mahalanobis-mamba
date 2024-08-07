import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def build_rotation_2d(r, epsilon=1e-8):
    n_w = r.shape[0]
    norms = torch.norm(r, dim=1, keepdim=True)
    r = r / (norms + epsilon)
    angles = torch.atan2(r[:, 0], r[:, 1])
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    rotation = torch.zeros(n_w, 2, 2, device=r.device, dtype=r.dtype)
    rotation[:, 0, 0] = cos
    rotation[:, 0, 1] = -sin
    rotation[:, 1, 0] = sin
    rotation[:, 1, 1] = cos
    
    return rotation

class GaussModel(nn.Module):
    def __init__(self, w_s, axis, device=None, dtype=None):
        super().__init__()
        self.w_s = w_s
        self.n_w = self.w_s * self.w_s

        self.scale = nn.Parameter(torch.zeros((self.n_w, 2), device=device, dtype=torch.float32), requires_grad=True)

        _rotation = torch.zeros((self.n_w, 2), device=device, dtype=torch.float32)
        _rotation[:, 0] = 1
        self.rotation = nn.Parameter(_rotation, requires_grad=True)
        self.rotation._no_weight_decay = True

        _mean = torch.zeros((self.n_w, 2), device=device, dtype=torch.float32)
        _mean[:, 0] = 0
        _mean[:, 1] = 0
        self.mean = nn.Parameter(_mean, requires_grad=True)
        self.mean._no_weight_decay = True

        self.axis = axis

    def forward(self, x):
        B, L, D = x.shape
        cls_token_pos = L // 2
        cls_token = x[:, cls_token_pos:cls_token_pos + 1, :]
        x = torch.cat((x[:, :cls_token_pos, :], x[:, cls_token_pos + 1:, :]), dim=1)
        
        _, l, _ = x.shape
        H = W = int(math.sqrt(l))

        assert H % self.w_s == 0 and W % self.w_s == 0
        h_w = int(H // self.w_s)
        w_w = int(W // self.w_s)

        scale = torch.exp(self.scale)
        left = torch.zeros((self.n_w, 2, 2), device=x.device, dtype=torch.float32)
        left[:, 0, 0] = scale[:, 0]
        left[:, 1, 1] = scale[:, 1]

        right = build_rotation_2d(self.rotation)
        transform = left @ right
        cov = transform @ transform.transpose(-2, -1) # (n_w, 2, 2)
        inv_cov = torch.cholesky_inverse(torch.linalg.cholesky(cov))  # (n_w, 2, 2)

        grid_y, grid_x = torch.meshgrid(torch.arange(h_w, device=x.device, dtype=torch.float32),
                                        torch.arange(w_w, device=x.device, dtype=torch.float32))
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (h_w, w_w, 2)
        grid = grid.view(1, 1, h_w, w_w, 2).expand(B, self.n_w, -1, -1, -1) # (B, n_w, h_w, w_w, 2)

        mean = torch.exp(self.mean)
        mean_mean = torch.mean(mean, dim=1, keepdim=True)  # (n_w, 2)
        mean_std = torch.std(mean, dim=1, keepdim=True)
        mean = (mean - mean_mean) / (mean_std + 1e-5)
        mean = mean * (h_w // 2) + (w_w // 2)
        mean = torch.clip(mean, min=0, max=(h_w // 2))
        mean = mean.view(1, self.n_w, 1, 1, 2).expand(B, -1, h_w, w_w, -1)  # (B, n_w, h_w, w_w, 2)

        diff = grid - mean  # (B, n_w, h_w, w_w, 2)
        inv_cov = inv_cov.view(1, self.n_w, 1, 1, 2, 2).expand(B, -1, h_w, w_w, -1, -1)  # (B, n_w, h_w, w_w, 2, 2)
        mahalanobis = ((diff.view(B, self.n_w, h_w, w_w, 1, 2) @ inv_cov) @ diff.view(B, self.n_w, h_w, w_w, 2, 1)).view(B, self.n_w, h_w, w_w)  # (B, n_w, h_w, w_w)
        weights = F.sigmoid(torch.exp(-0.5 * mahalanobis)).view(B, self.n_w, h_w * w_w) # (B, n_w, h_w * w_w)
        
        if self.axis == "out":
            axis = 1
        elif self.axis == "in":
            axis = 2
        else:
            return NotImplemented

        weights, weight_indices = torch.sort(weights, dim=axis, descending=True)
        weights = weights.view(B, self.n_w, h_w * w_w, 1).expand(-1, -1, -1, D) # (B, n_w, h_w * w_w, D)

        x = x.view(B, H, W, D)
        n_w_sqrt = int(self.n_w ** 0.5)
        x = x.view(B, n_w_sqrt, h_w, n_w_sqrt, w_w, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, self.n_w, h_w * w_w, D)
        
        weight_indices = weight_indices.view(B, self.n_w, h_w * w_w, 1).expand(-1, -1, -1, D)
        x = torch.gather(x, axis, weight_indices)
        
        x_new = x * weights  # (B, n_w, h_w * w_w, D)

        x_new = x_new.view(B, n_w_sqrt, n_w_sqrt, h_w, w_w, D)
        x_new = x_new.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_new = x_new.view(B, H, W, D)
        x_new = x_new.view(B, l, D)
        x_new = torch.cat((x_new[:, :cls_token_pos, :], cls_token, x_new[:, cls_token_pos:, :]), dim=1)
        assert (B, L, D) == x_new.shape
        
        return x_new

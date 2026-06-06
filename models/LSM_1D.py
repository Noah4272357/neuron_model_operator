"""
@author: Haixu Wu
"""
import torch.nn.functional as F
import torch.nn as nn
import torch
import math


################################################################
# Multiscale modules 1D
################################################################
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = x2.size(-1) - x1.size(-1)
        if diff > 0:
            x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        elif diff < 0:
            crop_left = (-diff) // 2
            x1 = x1[..., crop_left:crop_left + x2.size(-1)]

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


################################################################
# Patchify and Neural Spectral Block
################################################################
class NeuralSpectralBlock2d(nn.Module):
    def __init__(self, width, num_basis, patch_size=3, num_token=4):
        super(NeuralSpectralBlock2d, self).__init__()
        self.patch_size = patch_size
        self.width = width
        self.num_basis = num_basis

        # basis
        self.register_buffer(
            "modes_list",
            (1.0 / float(num_basis)) * torch.arange(num_basis, dtype=torch.float)
        )
        self.weights = nn.Parameter(
            (1 / (width)) * torch.rand(width, self.num_basis * 2, dtype=torch.float))
        # latent
        self.head = 8
        self.num_token = num_token
        self.latent = nn.Parameter(
            (1 / (width)) * torch.rand(self.head, self.num_token, width // self.head, dtype=torch.float))
        self.encoder_attn = nn.Conv1d(self.width, self.width * 2, kernel_size=1, stride=1)
        self.decoder_attn = nn.Conv1d(self.width, self.width, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=-1)

    def self_attn(self, q, k, v):
        # q,k,v: B H L C/H
        attn = self.softmax(torch.einsum("bhlc,bhsc->bhls", q, k))
        return torch.einsum("bhls,bhsc->bhlc", attn, v)

    def latent_encoder_attn(self, x):
        # x: B C H W
        B, C, H = x.shape
        L = H
        latent_token = self.latent.unsqueeze(0).expand(B, -1, -1, -1)
        x_tmp = self.encoder_attn(x).reshape(B, C * 2, -1).permute(0, 2, 1) \
            .reshape(B, L, self.head, C // self.head, 2).permute(4, 0, 2, 1, 3)
        latent_token = self.self_attn(latent_token, x_tmp[0], x_tmp[1]) + latent_token
        latent_token = latent_token.permute(0, 1, 3, 2).reshape(B, C, self.num_token)
        return latent_token

    def latent_decoder_attn(self, x, latent_token):
        # x: B C L
        x_init = x
        B, C, H = x.shape
        L = H
        latent_token = latent_token.reshape(B, self.head, C // self.head, self.num_token).permute(0, 1, 3, 2)
        x_tmp = self.decoder_attn(x).reshape(B, C, -1).permute(0, 2, 1) \
            .reshape(B, L, self.head, C // self.head).permute(0, 2, 1, 3)
        x = self.self_attn(x_tmp, latent_token, latent_token)
        x = x.permute(0, 1, 3, 2).reshape(B, C, H) + x_init  # B H L C/H
        return x

    def apply_basis_projection(self, x):
        # x: B C N
        modes = self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi
        sin_weights, cos_weights = self.weights.chunk(2, dim=-1)
        return (
            torch.einsum("bilm,im->bil", torch.sin(modes), sin_weights)
            + torch.einsum("bilm,im->bil", torch.cos(modes), cos_weights)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bilm,im->bil", input, weights)

    def forward(self, x):
        B, C, H = x.shape
        # print(x.shape)
        # patchify
        x = x.reshape(B, C, H // self.patch_size, self.patch_size) \
            .permute(0, 3, 1, 2) \
            .reshape(B * (H // self.patch_size), C, self.patch_size)
        # Neural Spectral
        # (1) encoder
        latent_token = self.latent_encoder_attn(x)
        # (2) transition
        latent_token = self.apply_basis_projection(latent_token) + latent_token
        # (3) decoder
        x = self.latent_decoder_attn(x, latent_token)
        # de-patchify
        x = x.reshape(B, H // self.patch_size, C, self.patch_size).permute(0, 2, 1, 3) \
            .reshape(B, C, H)
        return x


class LSM1d(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        width=64,
        lift_dim=128,
        num_token=8,
        num_basis=16,
        patch_size=1,
        padding=0,
        bilinear=True,
    ):
        super(LSM1d, self).__init__()
        patch_size = self._parse_int_value(patch_size)
        padding = self._parse_int_value(padding)
        # multiscale modules
        self.inc = DoubleConv(width, width)
        self.down1 = Down(width, width * 2)
        self.down2 = Down(width * 2, width * 4)
        self.down3 = Down(width * 4, width * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(width * 8, width * 16 // factor)
        self.up1 = Up(width * 16, width * 8 // factor, bilinear)
        self.up2 = Up(width * 8, width * 4 // factor, bilinear)
        self.up3 = Up(width * 4, width * 2 // factor, bilinear)
        self.up4 = Up(width * 2, width, bilinear)
        self.outc = OutConv(width, width)
        # Patchified Neural Spectral Blocks
        self.process1 = NeuralSpectralBlock2d(width, num_basis, patch_size, num_token)
        self.process2 = NeuralSpectralBlock2d(width * 2, num_basis, patch_size, num_token)
        self.process3 = NeuralSpectralBlock2d(width * 4, num_basis, patch_size, num_token)
        self.process4 = NeuralSpectralBlock2d(width * 8, num_basis, patch_size, num_token)
        self.process5 = NeuralSpectralBlock2d(width * 16 // factor, num_basis, patch_size, num_token)
        # projectors
        self.padding = padding
        self.fc0 = nn.Linear(in_channels, width)
        self.fc1 = nn.Linear(width, lift_dim)
        self.fc2 = nn.Linear(lift_dim, out_channels)

    @staticmethod
    def _parse_int_value(value):
        if isinstance(value, str):
            value = value.split(",")[0]
        return int(value)

    def forward(self, x, grid):
        if x.dim() == 2:
            x = torch.stack((x, grid), dim=-1)
        elif x.dim() == 3:
            if grid.dim() == 2:
                grid = grid.unsqueeze(-1)
            x = torch.cat((x, grid), dim=-1)
        else:
            raise ValueError(f"Expected x to have shape [B, S] or [B, S, C], got {tuple(x.shape)}.")

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        if self.padding:
            x = F.pad(x, [0, self.padding])

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(self.process5(x5), self.process4(x4))
        x = self.up2(x, self.process3(x3))
        x = self.up3(x, self.process2(x2))
        x = self.up4(x, self.process1(x1))
        x = self.outc(x)

        if self.padding:
            x = x[..., :-self.padding]
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.squeeze(-1)


def test_lsm1d_forward_output_shape():
    model = LSM1d(
        in_channels=2,
        out_channels=1,
        width=8,
        num_token=4,
        num_basis=6,
        patch_size=1,
        padding=0,
    ).eval()

    batch_size = 2
    seq_len = 5000

    x = torch.randn(batch_size, seq_len)
    grid = torch.linspace(0, 1, seq_len).repeat(batch_size, 1)

    with torch.no_grad():
        y = model(x, grid)

    assert y.shape == (batch_size, seq_len)

    x = x.unsqueeze(-1)
    with torch.no_grad():
        y = model(x, grid)

    assert y.shape == (batch_size, seq_len)
    print(y.shape)

if __name__=="__main__":
    test_lsm1d_forward_output_shape()

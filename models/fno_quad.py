import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat
        )
        out_ft[:, :, : self.modes] = self.compl_mul1d(x_ft[:, :, : self.modes], self.weights1)
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNOBlock1d(nn.Module):
    def __init__(self, modes, width):
        super(FNOBlock1d, self).__init__()
        self.modes = modes
        self.width = width
        self.conv = SpectralConv1d(self.width, self.width, self.modes)
        self.w = nn.Conv1d(self.width, self.width, 1)

    def forward(self, x):
        _x = x
        x1 = self.conv(x)
        x2 = self.w(x)
        x = F.gelu(x1 + x2)
        return _x + x


class FNOQuad1d(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        embed_dim=32,
        modes=16,
        width=64,
        lift_dim=128,
        num_blocks=4,
    ):
        super(FNOQuad1d, self).__init__()
        self.modes = modes
        self.width = width
        self.padding = 2
        self.num_blocks = num_blocks
        self.fc0 = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.width),
        )
        self.blocks = nn.ModuleList([FNOBlock1d(self.modes, self.width) for _ in range(self.num_blocks)])
        self.fc1 = nn.Sequential(
            nn.Linear(self.width, lift_dim),
            nn.GELU(),
            nn.Linear(lift_dim, self.width),
            nn.GELU(),
            nn.Linear(self.width, out_channels),
        )
        # Trainable quadrature constant.
        self.quadrature_bias = nn.Parameter(torch.zeros(1))

    def _fft_quadrature(self, y, grid):
        # Integral from zero-frequency FFT coefficient: sum(y) = Re(FFT(y)[0]).
        y_ft = torch.fft.fft(y, dim=-1)
        y_sum = y_ft.real[..., 0]
        dx = (grid[..., 1:] - grid[..., :-1]).mean(dim=-1)
        return y_sum * dx

    def forward(self, x, grid):
        x = torch.stack((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = F.pad(x, [0, self.padding])

        for block in self.blocks:
            x = block(x)

        x = x[..., :-self.padding]
        x = x.permute(0, 2, 1)
        y = self.fc1(x).squeeze(-1)

        quadrature = self._fft_quadrature(y, grid) + self.quadrature_bias
        return y + quadrature.unsqueeze(-1)

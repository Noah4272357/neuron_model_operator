import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

try:
    from .fno import FNOBlock1d
except ImportError:
    from fno import FNOBlock1d


class NeuralODE1d(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        embed_dim=32,
        modes=16,
        width=64,
        lift_dim=128,
        num_blocks=4,
        ode_steps=None,
        t0=0.0,
        t1=1.0,
        ode_method="rk4",
        rtol=1e-4,
        atol=1e-5,
    ):
        super(NeuralODE1d, self).__init__()

        """
        Neural ODE version of FNO1d.

        The encoder and decoder match FNO1d. Instead of applying a stack of
        discrete FNO blocks, this model uses one FNOBlock1d as the ODE function
        and integrates the latent state from t0 to t1.

        input shape: (batchsize, x=s)
        grid shape: (batchsize, x=s)
        output shape: (batchsize, x=s, c=out_channels)
        """

        self.modes = modes
        self.width = width
        self.padding = 2
        self.num_blocks = num_blocks
        self.ode_steps = ode_steps if ode_steps is not None else num_blocks
        self.t0 = t0
        self.t1 = t1
        self.ode_method = ode_method
        self.rtol = rtol
        self.atol = atol

        self.fc0 = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, self.width),
        )

        self.ode_func = FNOBlock1d(self.modes, self.width)

        self.fc1 = nn.Sequential(
            nn.Linear(self.width, lift_dim),
            nn.GELU(),
            nn.Linear(lift_dim, self.width),
            nn.GELU(),
            nn.Linear(self.width, out_channels),
        )

    def _ode_rhs(self, t, x):
        return self.ode_func(x)

    def _integrate(self, x):
        if self.ode_steps < 1:
            raise ValueError("ode_steps must be at least 1")

        t = torch.linspace(
            self.t0,
            self.t1,
            self.ode_steps + 1,
            device=x.device,
            dtype=x.dtype,
        )
        return odeint(
            self._ode_rhs,
            x,
            t,
            method=self.ode_method,
            rtol=self.rtol,
            atol=self.atol,
        )[-1]

    def forward(self, x, grid):
        x = torch.stack((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x = F.pad(x, [0, self.padding])
        x = self._integrate(x)
        x = x[..., :-self.padding]

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    batch_size = 2
    spatial_size = 1000
    model = NeuralODE1d(
        in_channels=2,
        out_channels=1,
        embed_dim=8,
        modes=4,
        width=8,
        lift_dim=16,
        num_blocks=2,
    )
    x = torch.randn(batch_size, spatial_size)
    grid = torch.linspace(0, 1, spatial_size).repeat(batch_size, 1)
    y = model(x, grid)

    expected_shape = (batch_size, spatial_size, 1)
    assert y.shape == expected_shape, f"Expected {expected_shape}, got {tuple(y.shape)}"
    print(f"NeuralODE1d test passed. output shape: {tuple(y.shape)}")

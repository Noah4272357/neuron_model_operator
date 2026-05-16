import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes 
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNOBlock1d(nn.Module):
    def __init__(self, modes, width):
        super(FNOBlock1d,self).__init__()
        self.modes = modes
        self.width = width
        
        self.conv = SpectralConv1d(self.width, self.width, self.modes)
        self.w = nn.Conv1d(self.width, self.width, 1)
        
    def forward(self,x):
        _x = x
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = F.gelu(x)
        return _x+x


class SpikeBasedNO(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        embed_dim=32,
        modes=16,
        width=64,
        lift_dim=128,
        num_blocks=4,
        shape_width=20,
    ):
        super(SpikeBasedNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        if out_channels != 1:
            raise ValueError("SpikeBasedNO expects out_channels=1.")
        if shape_width < 1:
            raise ValueError("shape_width must be a positive integer.")

        self.modes = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.num_blocks=num_blocks
        self.shape_width = shape_width
        self.fc0 = nn.Sequential(nn.Linear(in_channels, embed_dim), # input channel is 2: (a(x), x)
                                nn.GELU(),
                                nn.Linear(embed_dim,self.width)
                                )
        self.blocks = nn.ModuleList([FNOBlock1d(self.modes,self.width) for _ in range(self.num_blocks)])  
        
        
        self.fc1 = nn.Sequential(nn.Linear(self.width, lift_dim),
                                 nn.GELU(),
                                nn.Linear(lift_dim, self.width),
                                 nn.GELU(),
                                nn.Linear(self.width,out_channels))
        impulse_shape = torch.zeros(shape_width)
        impulse_shape[shape_width // 2] = 1.0
        self.impulse_shape = nn.Parameter(impulse_shape)
        

    def forward(self, x, grid):
        input_features = torch.stack((x, grid),dim=-1)

        x = self.fc0(input_features)
        x = x.permute(0, 2, 1)
        
        x = F.pad(x, [0, self.padding]) # pad the domain if input is non-periodic

        for block in self.blocks:
            x=block(x)
        
        x = x[..., :-self.padding]
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = torch.sigmoid(x)

        batch_size = x.shape[0]
        left_padding = (self.shape_width - 1) // 2
        right_padding = self.shape_width - 1 - left_padding
        x = x.permute(0, 2, 1).reshape(1, batch_size, -1)
        x = F.pad(x, [left_padding, right_padding])
        impulse_shape = self.impulse_shape.reshape(1, 1, self.shape_width).repeat(batch_size, 1, 1)
        x = F.conv1d(x, impulse_shape, groups=batch_size)
        x = x.reshape(batch_size, 1, -1).permute(0, 2, 1)
        return x



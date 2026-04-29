"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet Neural Operator for solving 
   parametric partialdifferential equations in computational mechanics problems.
   
-- This code is for 1-D Burger's equation (time-independent problem).
"""

import pywt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1D, IDWT1D
torch.manual_seed(0)
np.random.seed(0)

#def get_model(pde_name,width=64,level=8):
#    if pde_name.endswith('BBP'):
#        model = WNO1d(width, level, seq_len=3000)
#    else:
#        raise NotImplementedError('PDE not implemented')
#    return model
""" Def: 1d Wavelet layer """
class WaveConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, level, seq_len, device='cuda'):
        super(WaveConv1d, self).__init__()

        """
        1D Wavelet layer. It does Wavelet Transform, linear transform, and
        Inverse Wavelet Transform.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level  
        self.dwt_ = DWT1D(wave='db6', J=self.level, mode='symmetric').to(device)
        test_input = torch.zeros(1,1,seq_len).to(device)
        self.mode_data, _ = self.dwt_(test_input)
        self.modes1 = self.mode_data.shape[-1]
        
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))

    # Convolution
    def mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute single tree Discrete Wavelet coefficients using some wavelet     
        dwt = DWT1D(wave='db6', J=self.level, mode='symmetric').to(x.device)
        x_ft, x_coeff = dwt(x)
        
        # Multiply the final low pass and high pass coefficients
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-1],  device=x.device)
        out_ft = self.mul1d(x_ft, self.weights1)
        x_coeff[-1] = self.mul1d(x_coeff[-1], self.weights2)
        
        # Reconstruct the signal
        idwt = IDWT1D(wave='db6', mode='symmetric').to(x.device)
        x = idwt((out_ft, x_coeff))        
        return x

""" The forward operation """

class WNO1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, level, seq_len):
        super(WNO1dBlock, self).__init__()
        self.conv = WaveConv1d(in_channels, out_channels, level, seq_len)
        self.w = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = F.gelu(x)
        return x

class WNO1d(nn.Module):
    def __init__(self, width, level, seq_len, num_blocks=4):
        super(WNO1d, self).__init__()

        """
        The WNO network. It contains num_blocks layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. num_blocks layers of the integral operators v(+1) = g(K(.) + W)(v).
            W is defined by self.w; K is defined by self.conv .
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.level = level
        self.width = width
        self.padding = 2 # pad the domain when required
        self.num_blocks = num_blocks
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.blocks = nn.ModuleList([
            WNO1dBlock(self.width, self.width, self.level, seq_len)
            for _ in range(self.num_blocks)
        ])

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, grid):

        x = torch.stack((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) 

        for block in self.blocks:
            x = block(x)

        # x = x[..., :-self.padding] 
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    
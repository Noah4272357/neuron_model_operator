# coding=utf-8
import torch
import torch.nn as nn
from .model_utils import _get_act, _get_initializer,MLP

class DeepONet1D(nn.Module):
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size :int,
        query_dim: int ,
        in_channel_branch: int=1,
        out_channel: int=1,
        activation: str = "gelu",
        kernel_initializer: str = "Glorot normal"):
        super().__init__()
        in_channel_branch=in_channel_branch*size
        in_channel_trunk=query_dim
        out_channel_branch=out_channel_trunk=128*out_channel
        layer_sizes=[64]*4
        
        activation_branch = self.activation_trunk = _get_act(activation)
        
        self.branch = MLP(in_channel_branch,out_channel_branch,layer_sizes, activation_branch, kernel_initializer)
        self.trunk = MLP(in_channel_trunk,out_channel_trunk,layer_sizes, self.activation_trunk, kernel_initializer)

        self.out_channel = out_channel
        self.query_dim=query_dim
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

    def forward(self, x, grid):
        grid=grid[0].unsqueeze(-1)
        # Branch net to encode the input function
        x = self.branch(x)
        # Trunk net to encode the domain of the output function
        grid = self.activation_trunk(self.trunk(grid))

        x=torch.mm(x,grid.transpose(0,1))
        #x = torch.einsum("bhi,rh->bri", x, grid)
        # Add bias
        x += self.b
        return x


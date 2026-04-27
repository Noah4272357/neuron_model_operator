from .model_utils import MLP
def get_model(pde_name):
    if pde_name.endswith('BBP'):
        in_channel = 2
        out_channel = 1
        layer_sizes = [64]+[128]*3+[512]*3+[128]*3+[64]*3
    else:
        raise NotImplementedError('PDE not implemented')
    
    return MLP(in_channel,out_channel,layer_sizes)


from .losses import *
def get_loss_func(loss_func_name):
    loss_func_dict = {
        'relative_l2': LpLoss(d=1,p=2),
        'relative_l4': LpLoss(d=1,p=4),
        'weighted_l2': wLpLoss(d=1,p=2)
    }
    if loss_func_name not in loss_func_dict:
        raise ValueError(f"Loss function '{loss_func_name} not found.")
    return loss_func_dict[loss_func_name]
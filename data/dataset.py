import numpy as np
import torch
import scipy.io as scio
from torch.utils.data import TensorDataset,DataLoader,random_split
from sklearn.preprocessing import StandardScaler


def get_dataset(dataset_name,ntrain=1000,seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    filepath_dict = {
        'multi_hh': '../neuron_data/hh_step_500.npz',
        'multi_izhikevich': '../neuron_data/izhikevich_step_500.npz'

    }

    assert dataset_name in filepath_dict, f"Task name {dataset_name} not found in filepath_dict"   
    filepath=filepath_dict[dataset_name]
    data=np.load(filepath)
    if dataset_name.startswith('inverse'):
        labels=data['I_ext']
        features=data['V']
    else:
        features=data['I_ext']
        labels=data['V']
    grids = data['time']
    ntest = len(grids)-ntrain
    features,labels,grids=torch.from_numpy(features).float(),torch.from_numpy(labels).float(),torch.from_numpy(grids).float()     

    dataset=TensorDataset(features,labels,grids)
    train_dataset, test_dataset = random_split(
            dataset, 
            [ntrain, ntest],
            generator=torch.Generator().manual_seed(seed)  # 设置随机种子保证可重复性
        )
    
    return train_dataset,test_dataset

def get_dataloader(dataset_name,batch_size):
    train_dataset, test_dataset = get_dataset(dataset_name)
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    return train_loader,test_loader


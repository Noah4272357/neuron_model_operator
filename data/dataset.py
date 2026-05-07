import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


DATASET_REGISTRY = {
    'multi_hh': {
        'path': '../neuron_data/hh_step_500.npz',
        'feature_key': 'I_ext',
        'label_key': 'V',
        'grid_key': 'time',
    },
    'multi_izhikevich': {
        'path': '../neuron_data/izhikevich_step_500.npz',
        'feature_key': 'I_ext',
        'label_key': 'V',
        'grid_key': 'time',
    },
}


def _get_dataset_config(dataset_name):
    if dataset_name not in DATASET_REGISTRY:
        available = ', '.join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available datasets: {available}")
    return DATASET_REGISTRY[dataset_name]


def _load_npz_dataset(config):
    data = np.load(config['path'])
    try:
        features = data[config['feature_key']]
        labels = data[config['label_key']]
        grids = data[config['grid_key']]
    except KeyError as exc:
        available_keys = ', '.join(data.files)
        raise KeyError(f"Missing key {exc} in {config['path']}. Available keys: {available_keys}") from exc
    return features, labels, grids


def _to_tensor_dataset(features, labels, grids):
    tensors = (
        torch.from_numpy(features).float(),
        torch.from_numpy(labels).float(),
        torch.from_numpy(grids).float(),
    )
    return TensorDataset(*tensors)


def _split_dataset(dataset, ntrain, ntest, seed):
    total_size = len(dataset)
    if ntest is None:
        ntest = total_size - ntrain
    if ntrain < 0 or ntest < 0:
        raise ValueError("ntrain and ntest must be non-negative")
    if ntrain + ntest > total_size:
        raise ValueError(
            f"Requested ntrain + ntest = {ntrain + ntest}, but dataset only has {total_size} samples"
        )

    unused = total_size - ntrain - ntest
    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset, _ = random_split(dataset, [ntrain, ntest, unused], generator=generator)
    return train_dataset, test_dataset


def get_dataset(dataset_name, ntrain=1000, ntest=None, seed=42):
    """Load a dataset by name and return random train/test subsets.

    To add a new dataset, add one entry to DATASET_REGISTRY with its npz path
    and the keys for features, labels, and grids.
    """
    config = _get_dataset_config(dataset_name)
    features, labels, grids = _load_npz_dataset(config)
    dataset = _to_tensor_dataset(features, labels, grids)
    return _split_dataset(dataset, ntrain, ntest, seed)


def get_dataloader(dataset_name, batch_size, ntrain=1000, ntest=None, seed=42):
    train_dataset, test_dataset = get_dataset(dataset_name, ntrain=ntrain, ntest=ntest, seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

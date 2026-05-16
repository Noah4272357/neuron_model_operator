import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


DATASET_REGISTRY = {
    'hh_step': {
        'path': '../neuron_data/hh_step_500.npz',
        'feature_key': 'I_ext',
        'label_key': 'V',
        'grid_key': 'time',
    },
    'hh_poisson': {
        'path': '../neuron_data/hh_poisson_500.npz',
        'feature_key': 'I_ext',
        'label_key': 'V',
        'grid_key': 'time',
    },
    'hh_ou': {
        'path': '../neuron_data/hh_ou_500.npz',
        'feature_key': 'I_ext',
        'label_key': 'V',
        'grid_key': 'time',
    },
    'izhikevich_step': {
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


class NeuronDataset(Dataset):
    def __init__(
        self,
        features,
        labels,
        grids,
        label_min=None,
        label_max=None,
        normalize_labels=False,
    ):
        self.features = features.float()
        self.original_labels = labels.float()
        self.grids = grids.float()
        self.label_min = label_min
        self.label_max = label_max
        self.normalize_labels = normalize_labels

        if normalize_labels:
            label_range = self.label_max - self.label_min
            if torch.any(label_range == 0):
                raise ValueError("Cannot normalize labels with zero label range.")
            self.labels = self.normalize_label(self.original_labels)
        else:
            self.labels = self.original_labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.grids[idx]

    def normalize_label(self, labels):
        if self.label_min is None or self.label_max is None:
            return labels
        return (labels - self.label_min.to(labels.device)) / (
            self.label_max.to(labels.device) - self.label_min.to(labels.device)
        )

    def inverse_transform_label(self, labels):
        if self.label_min is None or self.label_max is None:
            return labels
        return labels * (
            self.label_max.to(labels.device) - self.label_min.to(labels.device)
        ) + self.label_min.to(labels.device)


def _split_indices(total_size, ntrain, ntest, seed):
    if ntest is None:
        ntest = total_size - ntrain
    if ntrain < 0 or ntest < 0:
        raise ValueError("ntrain and ntest must be non-negative")
    if ntrain + ntest > total_size:
        raise ValueError(
            f"Requested ntrain + ntest = {ntrain + ntest}, but dataset only has {total_size} samples"
        )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator)
    train_indices = indices[:ntrain]
    test_indices = indices[ntrain:ntrain + ntest]
    return train_indices, test_indices


def get_dataset(dataset_name, ntrain=1000, ntest=None, seed=42, normalize_labels=False):
    """Load a dataset by name and return random train/test subsets.

    To add a new dataset, add one entry to DATASET_REGISTRY with its npz path
    and the keys for features, labels, and grids.
    """
    config = _get_dataset_config(dataset_name)
    features, labels, grids = _load_npz_dataset(config)
    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels).float()
    grids = torch.from_numpy(grids).float()

    train_indices, test_indices = _split_indices(len(features), ntrain, ntest, seed)
    train_labels = labels[train_indices]
    label_min = train_labels.amin() if normalize_labels else None
    label_max = train_labels.amax() if normalize_labels else None

    train_dataset = NeuronDataset(
        features[train_indices],
        labels[train_indices],
        grids[train_indices],
        label_min=label_min,
        label_max=label_max,
        normalize_labels=normalize_labels,
    )
    test_dataset = NeuronDataset(
        features[test_indices],
        labels[test_indices],
        grids[test_indices],
        label_min=label_min,
        label_max=label_max,
        normalize_labels=normalize_labels,
    )
    return train_dataset, test_dataset


def get_dataloader(
    dataset_name,
    batch_size,
    ntrain=1000,
    ntest=None,
    seed=42,
    normalize_labels=False,
):
    train_dataset, test_dataset = get_dataset(
        dataset_name,
        ntrain=ntrain,
        ntest=ntest,
        seed=seed,
        normalize_labels=normalize_labels,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

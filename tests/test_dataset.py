import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from data.dataset import (
    DATASET_REGISTRY,
    _available_dataset_names,
    _get_dataset_config,
    _load_npz_dataset,
    get_dataset,
)


EXPECTED_PATHS = {
    "lif_step": "../neuron_data/lif_step_300.npz",
    "BBP_poisson": "../neuron_data/BBP_poisson_300.npz",
    "lif_poisson": "../neuron_data/lif_poisson_300.npz",
    "hh_step": "../neuron_data/hh_step_300.npz",
    "hh_poisson": "../neuron_data/new_hh_poisson_300.npz",
    "hh_ou": "../neuron_data/hh_ou_500.npz",
    "izhikevich_step": "../neuron_data/izhikevich_step_500.npz",
    "izhikevich_poisson": "../neuron_data/izhikevich_poisson_300.npz",
}


class DatasetRegistryTests(unittest.TestCase):
    def test_registry_contains_one_path_per_base_dataset(self):
        self.assertEqual(DATASET_REGISTRY, EXPECTED_PATHS)

    def test_forward_and_inverse_share_path_and_swap_keys(self):
        for name, path in EXPECTED_PATHS.items():
            with self.subTest(name=name):
                forward = _get_dataset_config(name)
                inverse = _get_dataset_config(f"inverse_{name}")

                self.assertEqual(forward["path"], path)
                self.assertEqual(inverse["path"], path)
                self.assertEqual(forward["feature_key"], "I_ext")
                self.assertEqual(forward["label_key"], "V")
                self.assertEqual(inverse["feature_key"], "V")
                self.assertEqual(inverse["label_key"], "I_ext")
                self.assertEqual(forward["grid_key"], "time")
                self.assertEqual(inverse["grid_key"], "time")

    def test_available_names_include_both_directions(self):
        expected = sorted(
            list(EXPECTED_PATHS)
            + [f"inverse_{name}" for name in EXPECTED_PATHS]
        )
        self.assertEqual(_available_dataset_names(), expected)

    def test_unknown_name_lists_available_datasets(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unknown dataset 'missing'.*Available datasets:",
        ) as context:
            _get_dataset_config("missing")

        for name in _available_dataset_names():
            self.assertIn(name, str(context.exception))


class DatasetLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "dataset.npz"
        self.current = np.arange(24, dtype=np.float32).reshape(4, 6)
        self.voltage = self.current + 100
        self.time = np.tile(
            np.arange(6, dtype=np.float32),
            (4, 1),
        )
        np.savez(
            self.path,
            I_ext=self.current,
            V=self.voltage,
            time=self.time,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_npz_loading_respects_direction(self):
        forward = {
            "path": self.path,
            "feature_key": "I_ext",
            "label_key": "V",
            "grid_key": "time",
        }
        inverse = {
            **forward,
            "feature_key": "V",
            "label_key": "I_ext",
        }

        features, labels, grids = _load_npz_dataset(forward)
        np.testing.assert_array_equal(features, self.current)
        np.testing.assert_array_equal(labels, self.voltage)
        np.testing.assert_array_equal(grids, self.time)

        features, labels, grids = _load_npz_dataset(inverse)
        np.testing.assert_array_equal(features, self.voltage)
        np.testing.assert_array_equal(labels, self.current)
        np.testing.assert_array_equal(grids, self.time)

    def test_get_dataset_preserves_split_and_normalization(self):
        with patch.dict(DATASET_REGISTRY, {"sample": str(self.path)}, clear=True):
            train, test = get_dataset(
                "sample",
                ntrain=3,
                ntest=1,
                seed=7,
                normalize_labels=True,
            )

        generator = torch.Generator().manual_seed(7)
        indices = torch.randperm(4, generator=generator)
        expected_train_labels = torch.from_numpy(
            self.voltage[indices[:3]]
        )

        self.assertEqual(len(train), 3)
        self.assertEqual(len(test), 1)
        self.assertEqual(train.label_min, expected_train_labels.amin())
        self.assertEqual(train.label_max, expected_train_labels.amax())
        torch.testing.assert_close(
            train.inverse_transform_label(train.labels),
            train.original_labels,
        )
        torch.testing.assert_close(
            test.inverse_transform_label(test.labels),
            test.original_labels,
        )


if __name__ == "__main__":
    unittest.main()

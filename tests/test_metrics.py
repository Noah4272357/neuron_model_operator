import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from utils.metrics import Spike_feature


def make_spike_features(output_features, label_features):
    spike_features = Spike_feature.__new__(Spike_feature)
    spike_features.output_features = output_features
    spike_features.label_features = label_features
    return spike_features


class SpikeFeatureISITests(unittest.TestCase):
    def test_isi_cv_excludes_samples_with_invalid_labels(self):
        spike_features = make_spike_features(
            {"ISI_CV": np.array([0.3, 0.8, np.nan, np.nan])},
            {"ISI_CV": np.array([0.2, np.nan, np.nan, 0.4])},
        )

        self.assertEqual(spike_features.ISI_CV_valid_count(), 2)
        self.assertAlmostEqual(spike_features.ISI_CV_error(), 1.5)

    def test_isi_values_excludes_samples_with_invalid_labels(self):
        spike_features = make_spike_features(
            {
                "ISI_values": np.array(
                    [
                        [10.2, 20.4],
                        [30.0, np.nan],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                    ]
                )
            },
            {
                "ISI_values": np.array(
                    [
                        [10.0, 20.0],
                        [np.nan, np.nan],
                        [40.0, np.nan],
                        [np.nan, np.nan],
                    ]
                )
            },
        )

        self.assertEqual(spike_features.ISI_values_valid_count(), 2)
        self.assertAlmostEqual(
            spike_features.ISI_values_accuracy(),
            1.0,
        )

    def test_valid_count_is_zero_when_all_labels_are_invalid(self):
        spike_features = make_spike_features(
            {"ISI_CV": np.array([0.2, np.nan])},
            {"ISI_CV": np.array([np.nan, np.nan])},
        )

        self.assertEqual(spike_features.ISI_CV_valid_count(), 0)
        self.assertEqual(spike_features.ISI_CV_error(), 0.0)


class SafeAverageTests(unittest.TestCase):
    def test_safe_average_returns_nan_for_no_valid_samples(self):
        from test import _safe_average

        self.assertTrue(math.isnan(_safe_average(0.0, 0)))
        self.assertEqual(_safe_average(3.0, 2), 1.5)


class ModelEvaluatorISITests(unittest.TestCase):
    def test_isi_metrics_use_valid_sample_counts(self):
        from test import ModelEvaluator

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, inputs, grid):
                return torch.zeros_like(inputs) + self.weight

        class FakeSpikeFeature:
            def __init__(self, outputs, labels):
                pass

            def spike_time_accuracy(self):
                return 0.0

            def mean_frequency_error(self):
                return 0.0

            def ISI_CV_error(self):
                return 1.5

            def ISI_CV_valid_count(self):
                return 2

            def time_to_first_spike_error(self):
                return 0.0

            def ISI_values_accuracy(self):
                return 1.0

            def ISI_values_valid_count(self):
                return 2

        inputs = torch.ones(4, 8)
        labels = torch.ones(4, 8)
        grid = torch.zeros(4, 8)
        evaluator = ModelEvaluator(
            TensorDataset(inputs, labels, grid),
            Model(),
        )

        with patch("test.Spike_feature", FakeSpikeFeature):
            performance = evaluator.calculate_performance(batch_size=4)

        self.assertEqual(performance["ISI_CV_error"], 0.75)
        self.assertEqual(performance["ISI_values_accuracy"], 0.5)
        self.assertEqual(performance["ISI_CV_valid_samples"], 2)
        self.assertEqual(performance["ISI_values_valid_samples"], 2)


if __name__ == "__main__":
    unittest.main()

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from utils import LpLoss, Spike_feature
from data.dataset import get_dataset

from models import get_model
import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module="efel"
)


def _safe_average(total: float, count: int) -> float:
    if count == 0:
        return float("nan")
    return float(total / count)


class ModelEvaluator:
    def __init__(self, dataset, model):
        """
        Args:
            dataset: The test dataset (PyTorch Dataset object).
            model: The trained model to evaluate.
        """
        self.dataset = dataset
        self.model = model
        self.relative_l2 = LpLoss(d=1, p=2, size_average=False)
        self.device = next(model.parameters()).device
        self.model.eval()

    def _to_original_scale(self, values):
        if hasattr(self.dataset, "inverse_transform_label"):
            return self.dataset.inverse_transform_label(values)
        return values

    def calculate_performance(self, batch_size=32):
        """Return per-sample averages for waveform and spike metrics."""
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        results = {
            "relative_l2": 0.0,
            "spike_time_accuracy": 0.0,
            "mean_frequency_error": 0.0,
            "ISI_CV_error": 0.0,
            "time_to_first_spike_error": 0.0,
            "ISI_values_accuracy": 0.0,
        }
        valid_counts = {
            "ISI_CV": 0,
            "ISI_values": 0,
        }
        total_samples = 0

        with torch.no_grad():
            for inputs, labels, grid in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                grid = grid.to(self.device)
                outputs = self.model(inputs, grid).reshape(labels.shape)
                outputs = self._to_original_scale(outputs)
                labels = self._to_original_scale(labels)

                results["relative_l2"] += self.relative_l2(
                    outputs, labels
                ).item()

                spike_features = Spike_feature(outputs, labels)
                results["spike_time_accuracy"] += (
                    spike_features.spike_time_accuracy()
                )
                results["mean_frequency_error"] += (
                    spike_features.mean_frequency_error()
                )
                results["ISI_CV_error"] += spike_features.ISI_CV_error()
                valid_counts["ISI_CV"] += (
                    spike_features.ISI_CV_valid_count()
                )
                results["time_to_first_spike_error"] += (
                    spike_features.time_to_first_spike_error()
                )
                results["ISI_values_accuracy"] += (
                    spike_features.ISI_values_accuracy()
                )
                valid_counts["ISI_values"] += (
                    spike_features.ISI_values_valid_count()
                )

                total_samples += labels.size(0)

        if total_samples == 0:
            raise ValueError("Cannot evaluate an empty dataset.")

        performance = {
            name: float(score / total_samples)
            for name, score in results.items()
            if name not in {"ISI_CV_error", "ISI_values_accuracy"}
        }
        performance["ISI_CV_error"] = _safe_average(
            results["ISI_CV_error"],
            valid_counts["ISI_CV"],
        )
        performance["ISI_values_accuracy"] = _safe_average(
            results["ISI_values_accuracy"],
            valid_counts["ISI_values"],
        )
        performance["ISI_CV_valid_samples"] = valid_counts["ISI_CV"]
        performance["ISI_values_valid_samples"] = valid_counts["ISI_values"]
        return performance

    def visualize_results(self,fig_name='results.png',sample_size=5): 
        """Method 2: Randomly select samples and plot predictions."""
        # Randomly sample indices
        indices = np.random.choice(len(self.dataset), sample_size, replace=False)
        subset = Subset(self.dataset, indices)
        loader = DataLoader(subset, batch_size=sample_size)

        input, labels, grid = next(iter(loader))
        
        with torch.no_grad():
            outputs = self.model(input.to(self.device), grid.to(self.device))
            outputs = self._to_original_scale(outputs)
            labels = self._to_original_scale(labels.to(self.device)).cpu()

        # Plotting
        plt.figure(figsize=(20,8))
        for i in range(sample_size):
            plt.subplot(2, sample_size, i + 1)
            plt.plot(grid[i].cpu().numpy(),labels[i].cpu().numpy(),label = 'Ground Truth')
            plt.plot(grid[i].cpu().numpy(),outputs[i].cpu().numpy(),label = 'Prediction')
            plt.legend()
            plt.subplot(2, sample_size, i+1+sample_size)
            plt.plot(grid[i].cpu().numpy(),input[i].cpu().numpy(), label = 'Current')
            plt.legend()
        plt.tight_layout()
        plt.savefig(fig_name)

if __name__ == "__main__":
    models = ["FNO"]
    for model_name in models:
        config_path = os.path.join(
            "configs", f"{model_name}_config1.yaml"
        )
        with open(config_path, "r") as config_file:
            model_param = yaml.safe_load(config_file)
        model = get_model(model_name, **model_param)
        loss_func_name = "relative_l2"
        dataset_name_list = ["hh_step"]

        for dataset_name in dataset_name_list:
            resume_path = os.path.join(
                "checkpoints",
                (
                    f"{model_name}_config1_{dataset_name}_"
                    f"{loss_func_name}_last.pth.tar"
                ),
            )
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint["state_dict"])

            _, test_dataset = get_dataset(dataset_name)
            evaluator = ModelEvaluator(test_dataset, model)
            perf = evaluator.calculate_performance()
            print(f"{model_name}: {perf}")
            evaluator.visualize_results(
                fig_name=f"{model_name}_{dataset_name}_test_results.png"
            )

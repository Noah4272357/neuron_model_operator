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

    def visualize_results(
        self,
        fig_name="results.png",
        sample_size=5,
        seed=42,
    ):
        """Plot representative predictions in a publication-ready layout."""
        if sample_size < 1:
            raise ValueError("sample_size must be at least 1.")
        if sample_size > len(self.dataset):
            raise ValueError(
                "sample_size cannot exceed the number of dataset samples."
            )

        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.dataset), sample_size, replace=False)
        subset = Subset(self.dataset, indices)
        loader = DataLoader(subset, batch_size=sample_size, shuffle=False)

        inputs, labels, grid = next(iter(loader))

        with torch.no_grad():
            outputs = self.model(
                inputs.to(self.device),
                grid.to(self.device),
            ).reshape(labels.shape)
            outputs = self._to_original_scale(outputs).cpu()
            labels = self._to_original_scale(
                labels.to(self.device)
            ).cpu()

        # 7.0 in fits the text width of most two-column LaTeX templates.
        style = {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.1,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.dpi": 600,
        }
        with plt.rc_context(style):
            fig, axes = plt.subplots(
                sample_size,
                2,
                figsize=(7.0, 1.35 * sample_size + 0.45),
                sharex="col",
                squeeze=False,
                constrained_layout=True,
                gridspec_kw={"width_ratios": (1.6, 1.0)},
            )
            voltage_min = min(labels.min().item(), outputs.min().item())
            voltage_max = max(labels.max().item(), outputs.max().item())
            voltage_padding = 0.05 * (voltage_max - voltage_min)
            if voltage_padding == 0:
                voltage_padding = 0.05 * max(abs(voltage_min), 1.0)
            voltage_limits = (
                voltage_min - voltage_padding,
                voltage_max + voltage_padding,
            )

            for i, (voltage_ax, current_ax) in enumerate(axes):
                time = grid[i].detach().cpu().numpy().squeeze()
                target = labels[i].numpy().squeeze()
                prediction = outputs[i].numpy().squeeze()
                current = inputs[i].detach().cpu().numpy().squeeze()

                voltage_ax.plot(
                    time,
                    target,
                    color="#222222",
                    label="Ground truth",
                    zorder=2,
                )
                voltage_ax.plot(
                    time,
                    prediction,
                    color="#0072B2",
                    linestyle="--",
                    label="Prediction",
                    zorder=3,
                )
                voltage_ax.set_ylim(voltage_limits)
                current_ax.plot(time, current, color="#D55E00")

                voltage_ax.set_ylabel(r"$V$")
                current_ax.set_ylabel(r"$I_{\mathrm{ext}}$")
                voltage_ax.text(
                    0.02,
                    0.92,
                    f"({chr(97 + i)})",
                    transform=voltage_ax.transAxes,
                    ha="left",
                    va="top",
                    fontweight="bold",
                )

                for ax in (voltage_ax, current_ax):
                    ax.grid(
                        axis="y",
                        color="#D9D9D9",
                        linewidth=0.45,
                        alpha=0.7,
                    )
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.margins(x=0)

            axes[0, 0].set_title("Membrane potential")
            axes[0, 1].set_title("Applied current")
            axes[-1, 0].set_xlabel("Time")
            axes[-1, 1].set_xlabel("Time")

            handles, legend_labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(
                handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.065),
                ncol=2,
                frameon=False,
                handlelength=2.4,
            )
            fig.savefig(fig_name, bbox_inches="tight", facecolor="white")
            plt.close(fig)

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
            print(model_name, dataset_name)
            for key, value in perf.items():
                if not key.endswith("valid_samples"):
                    print(key, value)
            print()
            evaluator.visualize_results(
                fig_name=f"{model_name}_{dataset_name}_test_results.png"
            )

import multiprocessing as mp
import os

import efel
import numpy as np
import torch


__all__ = ["get_features", "spike_time_error"]


def _get_num_workers(batch_size: int, n_jobs: int | None) -> int:
    if n_jobs is None:
        return min(batch_size, os.cpu_count() or 1)
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1.")
    return min(batch_size, n_jobs)


def get_features(
    data: torch.Tensor,
    feature: str,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Extract an eFEL feature from batched voltage traces.

    Args:
        data: Voltage traces with shape ``(batch_size, seq_len)``.
        feature: eFEL feature name, for example ``"AP_amplitude"``.
        n_jobs: Number of worker processes to use. Defaults to all available
            CPU cores capped by batch size.

    Returns:
        A numpy array containing the requested feature for each batch item. Scalar
        features are returned as ``(batch_size,)``. Multi-value features are
        returned as ``(batch_size, num_values)`` and padded with NaN when eFEL
        returns different lengths across traces.
    """
    if not isinstance(data, torch.Tensor):
        raise TypeError("data must be a torch.Tensor.")
    if data.ndim != 2:
        raise ValueError("data must have shape (batch_size, seq_len).")
    if not isinstance(feature, str):
        raise TypeError("feature must be a string.")

    batch_size, seq_len = data.shape
    if batch_size == 0:
        raise ValueError("data must contain at least one trace.")
    if seq_len == 0:
        raise ValueError("data must contain at least one time step.")

    voltage = data.detach().cpu().numpy()
    time = np.arange(seq_len, dtype=np.float64) * 0.1
    stim_end = float(seq_len * 0.1)

    traces = [
        {
            "T": time,
            "V": voltage[i],
            "stim_start": [0.0],
            "stim_end": [stim_end],
        }
        for i in range(batch_size)
    ]

    num_workers = _get_num_workers(batch_size, n_jobs)
    if num_workers > 1:
        context = mp.get_context("fork")
        with context.Pool(processes=num_workers) as pool:
            feature_values = efel.get_feature_values(
                traces,
                [feature],
                parallel_map=pool.map,
            )
    else:
        feature_values = efel.get_feature_values(traces, [feature])

    values = []
    for result in feature_values:
        value = result.get(feature)
        if value is None or len(value) == 0:
            values.append(np.array([np.nan], dtype=np.float32))
        else:
            values.append(np.asarray(value, dtype=np.float32))

    max_len = max(len(value) for value in values)
    output = np.full((batch_size, max_len), np.nan, dtype=np.float32)
    for i, value in enumerate(values):
        output[i, : len(value)] = value

    if max_len == 1:
        return output.squeeze(1)
    return output


def spike_time_error(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    n_jobs: int | None = None,
) -> float:
    """Return average batch accuracy for predicted spike times.

    A label spike is counted as accurate when ``outputs`` has at least one spike
    within a 1 ms window centered on that label spike.
    """
    if not isinstance(outputs, torch.Tensor):
        raise TypeError("outputs must be a torch.Tensor.")
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor.")
    if outputs.ndim != 2 or labels.ndim != 2:
        raise ValueError("outputs and labels must have shape (batch_size, seq_len).")
    if outputs.shape[0] != labels.shape[0]:
        raise ValueError("outputs and labels must have the same batch size.")

    output_spikes = get_features(outputs, "peak_time", n_jobs=n_jobs)
    label_spikes = get_features(labels, "peak_time", n_jobs=n_jobs)

    if output_spikes.ndim == 1:
        output_spikes = output_spikes[:, np.newaxis]
    if label_spikes.ndim == 1:
        label_spikes = label_spikes[:, np.newaxis]

    accuracies = []
    for output_times, label_times in zip(output_spikes, label_spikes):
        output_times = output_times[~np.isnan(output_times)]
        label_times = label_times[~np.isnan(label_times)]

        if label_times.size == 0:
            accuracies.append(1.0 if output_times.size == 0 else 0.0)
            continue

        correct = 0
        for label_time in label_times:
            if np.any(np.abs(output_times - label_time) <= 0.5):
                correct += 1
        accuracies.append(correct / label_times.size)

    return float(sum(accuracies) / len(accuracies))

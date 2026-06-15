import efel
import numpy as np
import torch


__all__ = ["get_features", "Spike_feature"]


def get_features(
    data: torch.Tensor,
    features: list[str],
) -> dict[str, np.ndarray]:
    """Extract eFEL features from batched voltage traces.

    Args:
        data: Voltage traces with shape ``(batch_size, seq_len)``.
        features: List of eFEL feature names.

    Returns:
        A dictionary mapping each feature name to its batched values. Scalar
        features have shape ``(batch_size,)``. Multi-value features have shape
        ``(batch_size, num_values)`` and are padded with NaN when eFEL returns
        different lengths across traces.
    """
    if not isinstance(data, torch.Tensor):
        raise TypeError("data must be a torch.Tensor.")
    if data.ndim != 2:
        raise ValueError("data must have shape (batch_size, seq_len).")
    if not isinstance(features, list):
        raise TypeError("features must be a list.")
    if not features:
        raise ValueError("features must contain at least one feature name.")
    if not all(isinstance(feature, str) for feature in features):
        raise TypeError("all features must be strings.")

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

    feature_values = efel.get_feature_values(traces, features)
    return {
        feature: _batch_feature_values(feature_values, feature, batch_size)
        for feature in features
    }


def _batch_feature_values(
    feature_values: list[dict],
    feature: str,
    batch_size: int,
) -> np.ndarray:
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


class Spike_feature:
    """Compare spike-related eFEL features for model outputs and labels."""

    FEATURES = [
        "peak_time",
        "mean_frequency",
        "ISI_CV",
        "time_to_first_spike",
        "ISI_values",
    ]

    def __init__(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        if not isinstance(outputs, torch.Tensor):
            raise TypeError("outputs must be a torch.Tensor.")
        if not isinstance(labels, torch.Tensor):
            raise TypeError("labels must be a torch.Tensor.")
        if outputs.ndim != 2 or labels.ndim != 2:
            raise ValueError(
                "outputs and labels must have shape (batch_size, seq_len)."
            )
        if outputs.shape[0] != labels.shape[0]:
            raise ValueError("outputs and labels must have the same batch size.")

        self.output_features = get_features(outputs, self.FEATURES)
        self.label_features = get_features(labels, self.FEATURES)

    def spike_time_accuracy(self) -> float:
        """Return the sum of per-sample predicted spike-time accuracies."""
        return self._feature_accuracy("peak_time")

    def mean_frequency_error(self) -> float:
        """Return the sum of per-sample mean-frequency relative errors."""
        return self._relative_error("mean_frequency")

    def ISI_CV_error(self) -> float:
        """Return the ISI-CV error sum for samples with valid labels."""
        return self._relative_error("ISI_CV", skip_invalid_labels=True)

    def ISI_CV_valid_count(self) -> int:
        """Return the number of samples with a defined label ISI-CV."""
        return self._valid_label_count("ISI_CV")

    def time_to_first_spike_error(self) -> float:
        """Return the sum of per-sample first-spike-time relative errors."""
        return self._relative_error("time_to_first_spike")

    def ISI_values_accuracy(self) -> float:
        """Return the ISI accuracy sum for samples with valid labels."""
        return self._feature_accuracy(
            "ISI_values",
            skip_invalid_labels=True,
        )

    def ISI_values_valid_count(self) -> int:
        """Return the number of samples with defined label ISI values."""
        return self._valid_label_count("ISI_values")

    def _relative_error(
        self,
        feature: str,
        skip_invalid_labels: bool = False,
    ) -> float:
        output_values = _as_batched_sequences(
            self.output_features[feature]
        )
        label_values = _as_batched_sequences(self.label_features[feature])

        error_sum = 0.0
        for output_items, label_items in zip(output_values, label_values):
            output_items = output_items[~np.isnan(output_items)]
            label_items = label_items[~np.isnan(label_items)]

            if label_items.size == 0:
                if skip_invalid_labels:
                    continue
                if output_items.size > 0:
                    error_sum += 1.0
                continue

            if output_items.size == 0:
                error_sum += 1.0
                continue

            output_value = output_items[0]
            label_value = label_items[0]
            if label_value == 0:
                error_sum += abs(output_value)
            else:
                error_sum += (
                    abs(output_value - label_value) / abs(label_value)
                )

        return float(error_sum)

    def _feature_accuracy(
        self,
        feature: str,
        skip_invalid_labels: bool = False,
    ) -> float:
        output_values = _as_batched_sequences(self.output_features[feature])
        label_values = _as_batched_sequences(self.label_features[feature])

        accuracies = []
        for output_items, label_items in zip(output_values, label_values):
            output_items = output_items[~np.isnan(output_items)]
            label_items = label_items[~np.isnan(label_items)]

            if label_items.size == 0:
                if skip_invalid_labels:
                    continue
                accuracies.append(1.0 if output_items.size == 0 else 0.0)
                continue

            correct = sum(
                np.any(np.abs(output_items - label_item) <= 0.5)
                for label_item in label_items
            )
            accuracies.append(correct / label_items.size)

        return float(np.sum(accuracies))

    def _valid_label_count(self, feature: str) -> int:
        label_values = _as_batched_sequences(self.label_features[feature])
        return int(np.sum(np.any(~np.isnan(label_values), axis=1)))


def _as_batched_sequences(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return values[:, np.newaxis]
    return values

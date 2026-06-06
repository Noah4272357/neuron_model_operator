from importlib import import_module


_MODEL_REGISTRY = {
    "DeepONet": ("models.deeponet", "DeepONet1D"),
    "FNO": ("models.fno", "FNO1d"),
    "fno_quad": ("models.fno_quad", "FNOQuad1d"),
    "NeuralODE": ("models.neural_ode", "NeuralODE1d"),
    "WNO": ("models.wno", "WNO1d"),
    "LSM": ("models.LSM_1D", "LSM1d"),
    "SpikeBasedNO": ("models.spike_based_no", "SpikeBasedNO"),
    "FourierTransformer": ("models.fourier_transformer", "FourierTransformer1D"),
}

_CLASS_EXPORTS = {
    class_name: module_name
    for module_name, class_name in _MODEL_REGISTRY.values()
}

__all__ = [*_CLASS_EXPORTS, "get_model"]


def _load_class(module_name, class_name):
    return getattr(import_module(module_name), class_name)


def __getattr__(name):
    if name in _CLASS_EXPORTS:
        return _load_class(_CLASS_EXPORTS[name], name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_model(model_name, **kwargs):
    """
    Factory function to initialize a model by name.
    kwargs are passed from your YAML config file.
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Model {model_name} not found. "
            f"Available models: {list(_MODEL_REGISTRY.keys())}"
        )

    module_name, class_name = _MODEL_REGISTRY[model_name]
    return _load_class(module_name, class_name)(**kwargs)

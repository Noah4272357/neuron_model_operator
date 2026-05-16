from .deeponet import DeepONet1D
from .fno import FNO1d
from .neural_ode import NeuralODE1d
from .wno import WNO1d
from .LSM_1D import LSM1d
from .spike_based_no import SpikeBasedNO

def get_model(model_name, **kwargs):
    """
    Factory function to initialize a model by name.
    kwargs are passed from your YAML config file.
    """
    models_dict = {
        "DeepONet": DeepONet1D,
        'FNO': FNO1d,
        'NeuralODE': NeuralODE1d,
        'WNO': WNO1d,
        'LSM': LSM1d,
        'SpikeBasedNO': SpikeBasedNO
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Model {model_name} not found. "
                         f"Available models: {list(models_dict.keys())}")
    
    # Instantiate the model with hyperparameters from YAML
    return models_dict[model_name](**kwargs)

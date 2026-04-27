import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from utils import get_loss_func
from data.dataset import get_dataset

from models import get_model

class ModelEvaluator:
    def __init__(self, dataset, model, metrics):
        """
        Args:
            dataset: The test dataset (PyTorch Dataset object).
            model: The trained model to evaluate.
            metrics: A dictionary of callables, e.g., {"accuracy": accuracy_fn}.
            sample_size: Number of input to show in visualization.
        """
        self.dataset = dataset
        self.model = model
        self.metrics = metrics
        self.device = next(model.parameters()).device
        self.model.eval()

    def calculate_performance(self, batch_size=32):
        """Method 1: Run metrics across the entire test dataset."""
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        results = {name: 0.0 for name in self.metrics.keys()}
        total_samples = 0

        with torch.no_grad():
            for input, labels, grid in loader:
                input, labels, grid = input.to(self.device), labels.to(self.device), grid.to(self.device)
                outputs = self.model(input,grid).reshape(labels.shape)
                
                # Update each metric
                for name, metric_fn in self.metrics.items():
                    # Assumes metric_fn returns a sum or cumulative value
                    results[name] += metric_fn(outputs, labels)
                
                total_samples += labels.size(0)

        # Average the results
        final_performance = {name: score / len(loader) for name, score in results.items()}
        return final_performance

    def visualize_results(self,sample_size=5): 
        """Method 2: Randomly select samples and plot predictions."""
        # Randomly sample indices
        indices = np.random.choice(len(self.dataset), sample_size, replace=False)
        subset = Subset(self.dataset, indices)
        loader = DataLoader(subset, batch_size=sample_size)

        input, labels, grid = next(iter(loader))
        
        with torch.no_grad():
            outputs = self.model(input.to(self.device), grid.to(self.device))

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
        plt.savefig('FNO_multi_hh.png')

# --- Example Usage ---

model_name = 'FNO'
config_path = os.path.join("configs",f"{model_name}_config1.yaml")
with open(config_path, 'r') as f:
    model_param = yaml.safe_load(f)
model = get_model(model_name, **model_param)

resume_path = os.path.join("checkpoints",f"{model_name}_config1_best.pth.tar")
checkpoint = torch.load(resume_path)
model.load_state_dict(checkpoint['state_dict'])

_,test_dataset = get_dataset('multi_hh')     
evaluator = ModelEvaluator(test_dataset, model, {"relative_l2": get_loss_func("relative_l2")})
perf = evaluator.calculate_performance()
print(perf)
#evaluator.visualize_results()
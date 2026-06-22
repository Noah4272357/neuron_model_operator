# Neuron Model Operator

This project trains and evaluates neural operator models for neuron simulation
tasks. The training pipeline maps time-dependent input currents to membrane
voltage traces, and it also supports inverse datasets that swap the input and
target arrays.

## Project Structure

```text
.
├── configs/                 # YAML model hyperparameter files
│   ├── DeepONet_config1.yaml
│   ├── FNO_config1.yaml
│   ├── Fourier_Transformer_config1.yaml
│   └── WNO_config1.yaml
├── data/
│   ├── dataset.py           # Dataset registry, NPZ loading, train/test split
│   └── transforms.py        # Data transform utilities
├── models/
│   ├── __init__.py          # Model registry and get_model factory
│   ├── deeponet.py          # DeepONet implementation
│   ├── fno.py               # Fourier Neural Operator implementation
│   ├── fno_quad.py          # Quadratic FNO variant
│   ├── fourier_transformer.py
│   ├── LSM_1D.py
│   ├── mlp.py
│   ├── model_utils.py
│   ├── neural_ode.py
│   ├── spike_based_no.py
│   └── wno.py               # Wavelet Neural Operator implementation
├── tests/
│   ├── test_dataset.py      # Dataset registry/loading tests
│   └── test_metrics.py      # Spike metric and evaluator tests
├── utils/
│   ├── __init__.py          # Loss function registry
│   ├── losses.py            # Relative/weighted Lp losses
│   ├── metrics.py           # Spike feature metrics
│   └── wasserstein.py       # Soft Wasserstein loss
├── main.py                  # Training entry point
├── trainer.py               # Training loop, validation, early stopping
├── test.py                  # Evaluation and visualization script
├── run.sh                   # Batch experiment launcher
├── pyproject.toml           # Python project metadata and dependencies
├── uv.lock                  # Locked dependency versions
├── experiment.md            # Experiment notes
└── LICENSE
```

Generated runtime outputs are written to:

- `checkpoints/`: best and latest model checkpoints.
- `logs/`: TensorBoard event files.
- `*_test_results.png`: evaluation visualizations produced by `test.py`.

## Usage

### 1. Install dependencies

This repository is configured as a `uv` Python project and requires Python
3.11 or newer.

```bash
uv sync
```

If you do not use `uv`, create a Python 3.11+ environment and install the
dependencies listed in `pyproject.toml`.

### 2. Prepare datasets

Datasets are loaded from NPZ files outside this repository. By default,
`data/dataset.py` expects them in `../neuron_data/` relative to the project
root.

Supported dataset names are:

```text
lif_step
BBP_poisson
lif_poisson
hh_step
hh_poisson
hh_ou
izhikevich_step
izhikevich_poisson
```

Each NPZ file must contain these arrays:

- `I_ext`: external input current.
- `V`: membrane voltage.
- `time`: time grid.

Every dataset also has an inverse version. Prefix the dataset name with
`inverse_`, for example `inverse_hh_step`, to use `V` as the input and `I_ext`
as the target.

### 3. Train a model

Run `main.py` with a registered model name, dataset name, model config, and
loss function:

```bash
uv run python main.py \
  --model_name WNO \
  --dataset_name hh_step \
  --model_config WNO_config1 \
  --loss_func_name relative_l2 \
  --batch_size 32 \
  --epochs 500 \
  --lr 0.001
```

Available model names are:

```text
DeepONet
FNO
fno_quad
NeuralODE
WNO
LSM
SpikeBasedNO
Fourier_Transformer
```

Available loss names are:

```text
relative_l2
relative_l4
weighted_l2
soft_wasserstein_loss
```

The training script saves:

- Best checkpoint: `checkpoints/<run_name>_best.pth.tar`
- Last checkpoint: `checkpoints/<run_name>_last.pth.tar`
- TensorBoard logs: `logs/<run_name>/`

To view TensorBoard logs:

```bash
uv run tensorboard --logdir logs
```

### 4. Resume training

Pass a checkpoint path with `--resume_path`:

```bash
uv run python main.py \
  --model_name WNO \
  --dataset_name hh_step \
  --model_config WNO_config1 \
  --loss_func_name relative_l2 \
  --resume_path checkpoints/WNO_config1_hh_step_relative_l2_last.pth.tar
```

Early stopping can be enabled with:

```bash
uv run python main.py --early_stopping --patience 10
```

Label min-max normalization can be enabled with:

```bash
uv run python main.py --normalize_labels
```

### 5. Run batch experiments

`run.sh` loops over the configured model, loss, dataset, and config arrays.
Edit the arrays in the script to change the experiment grid, then run:

```bash
bash run.sh
```

### 6. Evaluate a checkpoint

`test.py` loads a trained checkpoint, computes waveform and spike-feature
metrics, and saves a prediction visualization.

```bash
uv run python test.py
```

The default evaluation block currently evaluates `FNO` on `hh_step` using:

```text
checkpoints/FNO_config1_hh_step_relative_l2_last.pth.tar
```

Edit the `models`, `dataset_name_list`, and checkpoint path logic in `test.py`
to evaluate other runs.

### 7. Run tests

```bash
uv run python -m unittest discover tests
```

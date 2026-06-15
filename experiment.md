# 4 Experiments

This section systematically evaluates neural operators for surrogate modeling of neuronal dynamics. We first compare several representative neural operators on classical neuron models under diverse stimulation protocols and neuroscientifically meaningful evaluation metrics. Based on these results, we identify the most suitable architecture and subsequently investigate its applicability to biophysically detailed Blue Brain neuron models.

## 4.1 Experimental Settings

Briefly summarize experimental details necessary for reproducibility.

Contents:

* Dataset generation
* Training and testing splits
* Hyperparameter settings
* Hardware and software environment
* Evaluation metrics

Goal:
Provide implementation details while avoiding repetition of previous sections.

---

## 4.2 Benchmark Study on Classical Neuron Models

This section evaluates neural operators on the Hodgkin–Huxley and Izhikevich models.

### 4.2.1 Voltage Reconstruction Accuracy

Contents:

* Relative L2 error
* Representative voltage traces
* Comparison between operators

Figures:

* Predicted versus reference voltage traces
* Quantitative comparison table

Goal:
Evaluate the ability of neural operators to reconstruct membrane potential dynamics.

---

### 4.2.2 Electrophysiological Fidelity

Contents:

* Spike timing accuracy
* Firing rate accuracy
* ISI distribution comparison
* Representative spike train visualizations

Figures:

* Spike raster comparisons
* ISI histograms
* Statistical summary tables

Goal:
Evaluate whether surrogate models preserve functionally relevant neuronal behaviors rather than merely matching voltage trajectories.

---

### 4.2.3 Computational Efficiency

Contents:

* Inference time
* Speedup over numerical simulation
* Accuracy-efficiency comparison

Goal:
Assess the practical utility of neural-operator surrogates.

---

### 4.2.4 Comparative Analysis

Contents:

* Overall comparison across all metrics
* Strengths and weaknesses of each operator
* Selection of the best-performing architecture

Goal:
Identify the most suitable neural operator for realistic neuronal surrogate modeling.

---

## 4.3 Surrogate Modeling of Blue Brain Neurons

This section investigates whether neural operators can scale from simplified neuron models to biophysically realistic neuronal simulations.

### 4.3.1 Voltage Reconstruction Accuracy

Contents:

* Representative voltage traces
* Quantitative trajectory errors
* Comparison with classical neuron models

Goal:
Evaluate whether high-fidelity reconstruction remains achievable in a substantially more complex dynamical system.

---

### 4.3.2 Electrophysiological Fidelity

Contents:

* Spike timing accuracy
* Firing rate accuracy
* ISI statistics
* Representative firing patterns

Goal:
Assess whether the surrogate preserves the electrophysiological characteristics of realistic neurons.

---

### 4.3.3 Computational Acceleration

Contents:

* Runtime comparison
* Speedup factor
* Scalability discussion

Goal:
Demonstrate the practical value of the surrogate for large-scale neuroscience workflows.

---

## 4.4 Discussion

Contents:

* Main findings from classical neuron benchmarks
* Transferability to realistic Blue Brain neurons
* Relationship between trajectory accuracy and electrophysiological fidelity
* Implications for future neural-operator-based surrogate modeling

Goal:
Summarize insights and connect experimental results to the broader objectives of computational neuroscience and operator learning.

# Variant Shift Detection via Multi-Dimensional Profiling and Graph Modeling
This repository provides an implementation for detecting trace variant shifts in business processes using multi-dimensional profiling, wavelet transforms, and graph-based community detection.

## Overview
The main goal is to reveal subtle variant shifts by:

   - Multi-dimensional profiling (activity sequences, transitions, and temporal dynamics using wavelets)

   - Building k-NN graphs based on Levenshtein and Euclidean distances

   - Detecting communities to identify clusters of similar behavior

   - Spotting anomalies using One-Class SVM

This technique helps detect fine-grained, localized process changes, supporting both real-time monitoring and retrospective audits.



## Installation
1. Clone this repository: 

```console
https://github.com/grimali/04CL_changing.git
cd AI4BPM

```
2. Create a virtual environment and install dependencies:
```console
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt


```


## Usage
Run a detection experiment
```console
python variant_experiment_optimization.py --ocel_path datasets/your_log.xes

```
This script will:

   - Generate structural and temporal profiles for each trace.

   - Build a k-NN graph and detect communities.

  - Compute structural node metrics.

  - Apply a One-Class SVM to detect anomalous variants.

  -  Save results in resultados_ai4bpm/.
### Hyperparameter Optimization

The pipeline uses DEAP for evolutionary optimization of:

   - k: Number of neighbors in k-NN

   - w: Wavelet resolution

   - r: Louvain resolution for community detection

   - ν and γ: SVM hyperparameters

# Variant Shift Detection via Multi-Dimensional Profiling and Graph Modeling
This repository provides an implementation for detecting trace variant shifts in business processes using multi-dimensional profiling, wavelet transforms, and graph-based community detection.

##Overview
This approach combines:

Multi-dimensional profiling: activities, transitions, and temporal dynamics (using discrete wavelet transforms)

k-NN graph modeling: similarity graphs based on Levenshtein and Euclidean distances

Community detection: Louvain clustering to find behavior clusters

Anomaly detection: One-Class SVM for spotting variant shifts

This enables fine-grained monitoring and retrospective analysis of process evolution.



## Installation
The required packages can be installed using the following command in a terminal:

```console
 pip install -r requirements.txt
```


## Usage
The code can be run using the following command:
```console
 python run test_sample.py
```

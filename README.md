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
The code can be run using the following command:
```console
 python run test_sample.py
```

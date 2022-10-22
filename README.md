# ModelKeeper

This repository contains the evvaluation artifacts of our NSDI '23 paper "[ModelKeeper: Accelerating DNN Training via Automated Training Warmup](https://symbioticlab.org/publications/files/modelkeeper:nsdi23/modelkeeper-nsdi23.pdf)".

**ModelKeeper is being merged as part of [FedScale](https://github.com/SymbioticLab/FedScale) and is actively maintained there. Please try it!**

# Overview

* [Getting Started](#getting-started)
* [Run Experiments](#run-experiments)
* [Repo Structure](#repo-structure)
* [Contact](#contact)

# Getting Started

Our ```install.sh``` will install the following automatically:

* Anaconda Package Manager
* CUDA 10.2

Note: if you prefer different versions of conda and CUDA, please check  comments in `install.sh` for details.

Run the following commands to install ModelKeeper. 

```
source install.sh 
pip install -e .
```

# Run Experiments

# Repo Structure

```
Repo Root
|---- modelkeeper   # Core implementation (e.g., Matcher).
|---- evals         # MK support for different training backends
    |---- ray_tune      # Ray experiments
    |---- nni           # Retiarii experiments
|---- examples      # Toy experiments of model transformation
```

# Notes
please consider to cite our paper if you use the code or data in your research project.
```bibtex
@inproceedings{modelkeeper-nsdi23,
  title={ModelKeeper: Accelerating DNN Training via Automated Training Warmup},
  author={Fan Lai and Yinwei Dai and Harsha V. Madhyastha and Mosharaf Chowdhury},
  booktitle={USENIX Symposium on Networked Systems Design and Implementation (NSDI)},
  year={2023}
}
```

# Contact
Fan Lai (fanlai@umich.edu) and Yinwei Dai (yinweid@princeton.edu).



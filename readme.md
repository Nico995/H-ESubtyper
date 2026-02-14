# H&E-Subtyper

## Overview

This repository contains an obfuscated implementation of a deep learning pipeline for computational pathology. The project focuses on developing reproducible machine learning workflows for extracting representations from large histological images and training predictive models on downstream clinical endpoints.

The codebase demonstrates the engineering and methodological components required to build scalable, reproducible, and modular machine learning systems operating on high-dimensional medical imaging data.

To protect ongoing research and unpublished work, critical elements of the pipeline have been intentionally obfuscated.

---

## Obfuscation Notice

Part of the repository has been obfuscated until publication

---

## Repository Structure

```
.                     # Root directory 
├── configs           # Configuration files controlling experiments, models, and pipeline behavior
├── environment.yaml
├── LICENSE.txt
├── .env              # Private env variables
├── readme.md
├── requirements.txt  # Minimal Python dependency list for pip-based installation
├── scripts           # Entry-point scripts for running training, evaluation, and pipeline workflows
        └── ...
└── src               # Core implementation of deep learning pipeline
    ├── __init__.py
    ├── data          # Datasets and Datamodules
        └── ...
    ├── models        # Model definitions and abstractions for feature-based prediction
        └── ...
    ├── split.py      # Dataset splitting logic for reproducible training and validation partitions
    ├── train.py      # Entry point for training models using configuration-driven experiment setup
    ├── eval.py       # Entry point for model inference and performance evaluation
    └── utils         # General utility functions supporting training, evaluation, and pipeline infrastructure
        └── ...
```


The project follows a configuration-driven design to enable reproducibility and separation between code and experimental parameters.

---

## Technical Highlights

* Modular and extensible pipeline architecture
* Support for large-scale feature-based learning workflows
* Configuration-based experiment management
* Reproducible training and evaluation procedures

---

## License

This repository is provided for inspection and educational purposes only. See LICENSE for details.


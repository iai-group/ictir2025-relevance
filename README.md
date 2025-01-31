# Effect of Training Data on Neural Retrieval

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains experiments for training a BERT model on two types of training sets:

  - **Shallow-based**: Many queries with few relevance judgments.
  - **Depth-based**: Few queries with many relevance judgments.

These training sets will be sampled from the MS MARCO and LongEval datasets.

Paper: TBA

## Training

For detailed instruction on how to generate the different training sets and fine-tune the models, see [Generate Training Sets](docs/generate_trainingsets.md) and [Fine-Tuning](docs/fine_tuning.md).

## Evaluation

For detailed evaluation methods, see [evaluation details](docs/evaluation.md).

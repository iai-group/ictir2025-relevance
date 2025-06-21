#  Impact of Shallow vs. Deep Relevance Judgments on BERT-based Reranking Models

<!-- [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) -->

This repository contains experiments for training a BERT model on two types of training sets:

  - **Shallow-based**: Many queries with few relevance judgments.
  - **Depth-based**: Few queries with many relevance judgments.

These training sets will be sampled from the MS MARCO (V1 and V2) and LongEval collections.

Paper: Impact of Shallow vs. Deep Relevance Judgments on BERT-based Reranking Models

## Training

For detailed instruction on how to generate the different training sets and fine-tune the models, see [Generate Training Sets](docs/generate_trainingsets.md) and [Fine-Tuning](docs/fine_tuning.md).

## Evaluation

For detailed evaluation methods, see [evaluation details](docs/evaluation.md).

## Team

* [Gabriel Iturra-Bocaz](https://giturra.cl/)
* [Danny Vo](https://www.linkedin.com/in/danny-vo-157bab295)
* [Petra Galuščáková](https://galuscakova.github.io/) 

## Contact
Please write to ```gabriel.e.iturrabocaz``` at ```uis.no``` for inquiries about the software or, feel free to open an issue.

## Citation

If you find our paper or code useful, please cite the paper:

```
@inproceedings{Iturra:2025:ICTIR,
  author = {Iturra-Bocaz, Gabriel and Vo, Danny and Petra Galuscakova},
  title = {Impact of Shallow vs. Deep Relevance Judgments on BERT-based Reranking Models},
  booktitle = {Proceedings of the 2025 International ACM SIGIR Conference on Innovative Concepts and Theories in Information Retrieval}
  series = {ICTIR '25},
  year = {2025}
}
```
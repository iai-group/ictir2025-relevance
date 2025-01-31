# Generating Trainingsets

## Overview

This guide outlines the process for making the trainingsets used to fine-tune a BERT model on sequence classification task.

## Environment Setup

Create a new environment:

```
conda create -n name python=3.10
conda activate name
```

Replace *name*  with the name you want to call the environment.

Install the requirements file for reranking (The reranking script shares some of the libraries needed):

```
pip install -r requirements_reranking.txt
```

## Running the script

Change into the `scripts` folder and run the command:

```
python bm25_negative_sampling.py
```

Before running make sure to rename the file paths and change the arguments in the `Create training sets` section at the end of the file to suit your needs.

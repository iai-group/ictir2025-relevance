# Fine-tune a BERT model on sequence classification

## Environment Setup

Create a new environment:

```
conda create -n name python=3.10
conda activate name
```

Replace *name*  with the name you want to call the environment.

Install the requirements file for fine-tuning:

```
pip install -r requirements_fine_tuning.txt
```

## Train the model

Make sure to check that the file paths are correct, this includes the file path to your trainingset and also for where the model and checkpoints will be saved.

```
# Configs
DATASET_PATH = "/path/to/trainingset"
MODEL_SAVE_PATH = "/save/path/for/model/name_of_model"
CHECKPOINT_PATH = "/save/path/for/checkpoint/name_of_model"
```

Run the training script by changing into the `code` folder and use:

```
python fine_tune.py
```

## Evaluation of model

Download the `trec_eval` software from the [TREC Eval GitHub](https://github.com/usnistgov/trec_eval) and follow there instructions to set this up.

Activate the same environment that you used to create the trainingsets (reranking requirements file).
Change into the `code` folder and run the script for reranking:

```
python rerank.py
```

Make sure to change the filepath to where your model is saved and also where you want the run file that the script will make to be saved.

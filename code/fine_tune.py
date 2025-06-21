"""Fine-tune a BERT model on sequence classification task."""

import logging
import os

# Specify the GPU ID to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch  # noqa: E402
from datasets import load_from_disk  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from transformers import (  # noqa: E402
    BertForSequenceClassification,
    BertTokenizer,
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

seed = 5

dataset = "shallow_based_754_1_1"
# Configs
DATASET_PATH = (
    f"../data/longeval/bm25/{seed}/{dataset}/"
)
MODEL_SAVE_PATH = (
    f"../data/results/models/longeval/bm25/one_to_one/{seed}/{dataset}/"
)
CHECKPOINT_PATH = (
    f"../data/results/checkpoints/longeval/bm25/one_to_one/{seed}/{dataset}/"
)


def tokenize_function(tokenizer, examples):
    """Tokenize the examples.

    Args:
        tokenizer: tokenizer to use.
        examples: examples to tokenize.
    """
    # Tokenize the query and document texts and return the inputs necessary
    return tokenizer(
        examples["query_text"],
        examples["doc_text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "f1": f1,
        "accuracy": acc,
    }


# Ensure the GPU (if available) is used for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset and tokenizer
dataset = load_from_disk(DATASET_PATH)
logging.info(f"Loaded {len(dataset)} examples from {DATASET_PATH}")

# Split the dataset into training and validation sets
logging.info("Splitting the dataset into training and validation sets...")
dataset_dict = dataset.train_test_split(test_size=0.2)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
logging.info("Tokenizing the training set...")
tokenized_train_dataset = dataset_dict["train"].map(
    lambda examples: tokenize_function(tokenizer, examples), batched=True
)
# Tokenize the validation set
logging.info("Tokenizing the validation set...")
tokenized_val_dataset = dataset_dict["test"].map(
    lambda examples: tokenize_function(tokenizer, examples), batched=True
)

# Rename the relevance column to labels
tokenized_train_dataset = tokenized_train_dataset.map(
    lambda e: {"labels": e["relevance"]}
)
tokenized_val_dataset = tokenized_val_dataset.map(
    lambda e: {"labels": e["relevance"]}
)

# Set format for PyTorch
tokenized_train_dataset.set_format(
    type="torch",
    columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
)
tokenized_val_dataset.set_format(
    type="torch",
    columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
)

# Initialize the model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)
# model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
# model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

# Define the training arguments
training_args = TrainingArguments(
    output_dir=CHECKPOINT_PATH,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="../data/logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,
    early_stopping_threshold=0.001,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)

logging.info("Training the model...")
model.to(device)
trainer.train()

# Save the model
logging.info("Saving the model and tokenizer...")
trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

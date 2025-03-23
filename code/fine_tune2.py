"""Fine-tune a BERT model on sequence classification task."""

import logging
import os

# Specify the GPU ID to use
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from t5_reranker import T5Reranker  # noqa: E402

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

from safetensors.torch import save_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)



seed = 92

dataset = "shallow_based_1250_2_4"
# Configs
DATASET_PATH = (
    f"../data/hf_datasets/msmarcov1/test/bm25/one_to_one/v2/{seed}/{dataset}/"
)
MODEL_SAVE_PATH = (
    f"../data/results/models/t5/bm25/one_to_one/{seed}/{dataset}/"
)
CHECKPOINT_PATH = (
    f"../data/results/checkpoints/t5/bm25/one_to_one/{seed}/{dataset}/"
)


def tokenize_function(tokenizer, examples):
    """Proper T5 input formatting"""
    combined = [
        f"Query: {q} Document: {d} </s>"  # Add T5 end-of-sequence token
        for q, d in zip(examples["query_text"], examples["doc_text"])
    ]
    
    return tokenizer(
        combined,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"  # Keep tensor output
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
# tokenizer = BertTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

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
print(tokenized_train_dataset[0])
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
    columns=["input_ids", "attention_mask", "labels"],
)
tokenized_val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

# Initialize the model
# model = BertForSequenceClassification.from_pretrained(
#     "bert-base-uncased", num_labels=2
# )
# model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
# model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
base_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")
model = T5Reranker(base_model)

# Define the training arguments
training_args = TrainingArguments(
    output_dir=CHECKPOINT_PATH,
    save_safetensors=False,  # Force Safetensors format
    save_total_limit=1,  # Only keep best model,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
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

# # Explicitly handle shared weights
# model.base_model.shared = model.base_model.encoder.embed_tokens
# model.base_model.decoder.embed_tokens = model.base_model.shared
# model.base_model.lm_head.weight = model.base_model.shared.weight

# # Save using Safetensors format
# save_model(
#     model,
#     os.path.join(MODEL_SAVE_PATH, "model.safetensors"),
#     metadata={"format": "pt", "architecture": "T5Reranker"}
# )

# # Save tokenizer and config separately
# tokenizer.save_pretrained(MODEL_SAVE_PATH)
# model.config.save_pretrained(MODEL_SAVE_PATH)

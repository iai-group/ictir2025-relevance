"""Evaluate model using Pyserini and TREC Eval."""

import csv
import json
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import ir_datasets  # noqa: E402
import torch  # noqa: E402
from pyserini.search.lucene import LuceneSearcher  # noqa: E402
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import logging as tf_logging  # noqa: E402

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Comment out if you want to see the warnings
tf_logging.set_verbosity_error()


# Ensure the GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_queries(query_file):
    """Load queries from a TSV file."""
    queries = {}
    with open(query_file, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        for parts in reader:
            if len(parts) == 2:
                query_id, text = parts
                queries[query_id] = text
            else:
                print(f"Skipping malformed line: {parts}")
    return queries


# Load the index
# searcher = LuceneSearcher.from_prebuilt_index("msmarco-v2-passage")
searcher = LuceneSearcher("../data/indexes/longeval_st_index/")

# Load model and tokenizer
model_path = "bert-base-uncased"
logging.info(f"Loading model and tokenizer from: {model_path}")
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# Load queries
# query_dataset = ir_datasets.load("msmarco-passage-v2/dev1")
query_dataset_path = "../data/collections/longeval/test-collection/A-Short-July/English/Queries/test07.tsv"

# MSMARCO
# queries = {}
# for query in query_dataset.queries_iter():
#     queries[query.query_id] = query.text

# Longeval
queries = load_queries(query_dataset_path)

logging.info(f"Loaded {len(queries)} queries.")

# Setting up the run file
run_file_path = "../data/results/runs/zero_shot_st_run.txt"
run_name = "zero-shot-st-longeval"

# Create the directory if it does not exist
directory = os.path.dirname(run_file_path)
os.makedirs(directory, exist_ok=True)

logging.info("Evaluating the model...")
with open(run_file_path, "w") as run_file:
    for query_id, query in queries.items():
        # Search for the query
        hits = searcher.search(query)

        # Rerank the hits
        reranked_docs = []
        for hit in hits[:100]:
            doc = searcher.doc(hit.docid)
            doc_text = doc.raw()
            # passage = json.loads(doc_text)["passage"]
            passage = json.loads(doc_text)["contents"]
            inputs = tokenizer.encode_plus(
                query,
                doc_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            )

            # Move each tensor in the inputs dictionary to the specified device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            score = probabilities[:, 1].item()
            reranked_docs.append((hit.docid, score))

        # Sort the documents by score for this query
        reranked_docs.sort(key=lambda x: x[1], reverse=True)

        # Write the reranked results to the run file
        for i, (docid, score) in enumerate(reranked_docs):
            run_file.write(f"{query_id} Q0 {docid} {i+1} {score} {run_name}\n")
logging.info("Model evaluation complete.")

logging.info(f"Run file saved to: {run_file_path}")

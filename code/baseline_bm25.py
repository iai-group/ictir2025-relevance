"""BM25 baseline for msmarco passage v2 ranking task."""

import logging

import ir_datasets
from pyserini.search.lucene import LuceneSearcher

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load the prebuilt index
searcher = LuceneSearcher.from_prebuilt_index("msmarco-v2-passage")

# Load queries
query_dataset = ir_datasets.load("msmarco-passage-v2/dev1")
queries = {}
for query in query_dataset.queries_iter():
    queries[query.query_id] = query.text
logging.info(f"Loaded {len(queries)} queries.")

# Setting up the run file for the BM25 baseline
run_file_path = "../data/results/runs/bm25_baseline_run.txt"
run_name = "bm25_baseline"

# Generate the run file
logging.info("Generating BM25 baseline...")
with open(run_file_path, "w") as run_file:
    for query_id, query in queries.items():
        # Search for the query
        hits = searcher.search(query)

        # Write the hits to the run file
        for i, hit in enumerate(hits[:100]):
            run_file.write(
                f"{query_id} Q0 {hit.docid} {i + 1} {hit.score} {run_name}\n"
            )

logging.info("BM25 baseline run file generated.")
logging.info(f"Run file saved to {run_file_path}")

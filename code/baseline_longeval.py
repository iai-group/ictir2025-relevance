"""BM25 baseline for longeval ranking task."""

import csv
import logging

from pyserini.search.lucene import LuceneSearcher

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


# Load the prebuilt index
searcher = LuceneSearcher("../data/indexes/longeval_st_index/")

# Load queries
query_dataset_path = "../data/collections/longeval/test-collection/A-Short-July/English/Queries/test07.tsv"
queries = load_queries(query_dataset_path)
logging.info(f"Loaded {len(queries)} queries.")

# Setting up the run file for the BM25 baseline
run_file_path = "../data/results/runs/longeval_baseline_st_run.txt"
run_name = "longeval_st-bm25-baseline"

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

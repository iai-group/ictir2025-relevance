import csv
import os
import random
from collections import defaultdict

from datasets import Dataset
from halo import Halo
from pyserini.search.lucene import LuceneSearcher


def parse_trec_document(file_path):
    """Parse a TREC formatted document."""
    with open(file_path, "r") as file:
        content = file.read()

    docs = {}
    for doc in content.split("<DOC>")[1:]:
        doc_id = doc.split("<DOCNO>")[1].split("</DOCNO>")[0].strip()
        text = doc.split("<TEXT>")[1].split("</TEXT>")[0].strip()
        docs[doc_id] = text
    return docs


def load_queries_file(query_file):
    """Load queries from a text file."""
    queries = {}
    with open(query_file, "r") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                query_id, _, english_text = parts
                queries[query_id] = english_text
            else:
                print(f"Skipping malformed line: {line.strip()}")
    return queries


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


def load_qrels(qrels_files):
    """Load qrels from multiple text files.

    Args:
        qrels_files (list): list of qrels file paths.

    Returns:
        dict: dictionary containing relevance judgments by query id.
    """
    qrels = defaultdict(list)
    for qrels_file in qrels_files:
        with open(qrels_file, "r") as file:
            for line in file:
                query_id, _, doc_id, relevance = line.strip().split()
                qrels[query_id].append((doc_id, int(relevance)))
    return qrels


def select_queries(qrels_by_query_id, num_queries, num_rels_per_query):
    """Select queries with the number of relevance judgments specified.

    Selects a specified number of queries with a given number of relevance
    judgments per query. Queries are selected randomly from the provided
    relevance judgments. Only queries with a relevance score of 1 or larger is
    considered.

    Args:
        qrels_by_query_id (dict): dictionary containing relevance judgments by
            query id.
        num_queries (int): number of queries to select.
        num_rels_per_query (int): number of relevant judgments per query.

    Returns:
        list: list of selected queries with the specified relevance criteria.
    """
    # Create a list of queries that have the required number of qrels
    eligible_queries = []
    for query, qrels in qrels_by_query_id.items():
        relevant_qrels = [qrel for qrel in qrels if qrel[1] >= 1]
        if len(relevant_qrels) >= num_rels_per_query:
            eligible_queries.append(query)

    # Randomly select queries from the list of eligible queries
    selected_queries = []
    while eligible_queries and len(selected_queries) < num_queries:
        query = random.choice(eligible_queries)
        selected_queries.append(query)
        eligible_queries.remove(query)

    # Throw an error if not enough queries are found
    if len(selected_queries) < num_queries:
        raise ValueError(
            f"Could not find {num_queries} queries with {num_rels_per_query}"
            f"relevance judgments."
        )

    return selected_queries


def create_shallow_training_set(
    qrels_by_query_id,
    query_text_map,
    num_queries,
    num_rels_per_query,
    seed=42,
    skip_top_k=50,
):
    """Generates a training set with BM25-based negative sampling."""

    random.seed(seed)
    training_set = []

    # Select queries with the specified number of relevance judgments
    selected_queries = select_queries(
        qrels_by_query_id, num_queries, num_rels_per_query
    )

    for qid in selected_queries:
        query_text = query_text_map[qid]
        positive_docs = random.sample(
            qrels_by_query_id[qid], num_rels_per_query
        )
        negative_docs = []

        # Retrieve documents using BM25 and filter out positive documents
        # Targets documents lower in the BM25 ranking
        hits = searcher.search(query_text, k=100 + len(positive_docs))
        for hit in hits[skip_top_k:]:
            if (
                hit.docid not in positive_docs
                and len(negative_docs) < num_rels_per_query
            ):
                if hit.docid is tuple:
                    print(hit.docid)
                    negative_docs.append(hit.docid[0])
                else:
                    negative_docs.append(hit.docid)

        # Add both positive and negative examples to the training set
        for pos_doc_id in positive_docs:
            doc_id = pos_doc_id[0]
            training_set.append((qid, doc_id, 1))

        for neg_doc_id in negative_docs:
            training_set.append((qid, neg_doc_id, 0))

    return training_set


def create_depth_training_set(
    qrels_by_query_id,
    query_text_map,
    seed=42,
    skip_top_k=50,
):
    """Generates a training set with BM25-based negative sampling."""

    random.seed(seed)
    training_set = []

    for qid, qrels in qrels_by_query_id.items():
        query_text = query_text_map[qid]
        positive_docs = [qrel for qrel in qrels if qrel[1] >= 1]
        negative_docs = []

        # Retrieve documents using BM25 and filter out positive documents
        # Targets documents lower in the BM25 ranking
        hits = searcher.search(query_text, k=100 + len(positive_docs))
        for hit in hits[skip_top_k:]:
            if hit.docid not in positive_docs and len(negative_docs) < len(
                positive_docs
            ):
                if hit.docid is tuple:
                    negative_docs.append(hit.docid[0])
                else:
                    negative_docs.append(hit.docid)

        # Add both positive and negative examples to the training set
        for pos_doc_id in positive_docs:
            doc_id = pos_doc_id[0]
            training_set.append((qid, doc_id, 1))

        for neg_doc_id in negative_docs:
            training_set.append((qid, neg_doc_id, 0))

    return training_set


def training_set_to_dataset(training_set, query_text_map, doc_text_map):
    """Convert the training set to a Huggingface dataset.

    Arg:
        training_set (list): list of tuples with training set information.
        query_text_map (dict): dictionary mapping query IDs to their texts.
        doc_text_map (dict): dictionary mapping document IDs to their texts.

    Returns:
        Dataset: Huggingface dataset with the training set information.
    """
    # Initialize lists to hold column data
    query_ids = []
    doc_ids = []
    query_texts = []
    doc_texts = []
    relevances = []

    # Populate the lists with data from the training set
    for query_id, doc_id, relevance in training_set:
        query_ids.append(query_id)
        doc_ids.append(doc_id)
        query_texts.append(query_text_map[query_id])
        doc_texts.append(doc_text_map[doc_id])
        relevances.append(relevance)

    # Create a dictionary that maps column names to data lists
    data_dict = {
        "query_id": query_ids,
        "doc_id": doc_ids,
        "query_text": query_texts,
        "doc_text": doc_texts,
        "relevance": relevances,
    }

    # Convert to Huggingface dataset
    dataset = Dataset.from_dict(data_dict)
    return dataset


if __name__ == "__main__":
    index_path = "../data/indexes/longeval_index/"
    corpus_path = "../data/collections/longeval/publish/English/Documents/Trec/"
    corpus_queries_path_1 = (
        "../data/collections/longeval/publish/English/Queries/heldout.tsv"
    )
    corpus_queries_path_2 = (
        "../data/collections/longeval/publish/French/Queries/train.tsv"
    )
    depth_query_file = "../data/wt-queries-mapping.txt"
    depth_qrels_file = "../data/annotations_wt.qrels"
    shallow_qrel_file_1 = "../data/collections/longeval/longeval-relevance-judgements/heldout-test.txt"
    shallow_qrel_file_2 = (
        "../data/collections/longeval/publish/French/Qrels/train.txt"
    )

    # Initialize BM25 searcher
    searcher = LuceneSearcher(index_path)

    spinner = Halo(text="Loading documents...", spinner="dots")
    spinner.start()
    # Load documents
    doc_text_map = {}
    for filename in os.listdir(corpus_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(corpus_path, filename)
            doc_text_map.update(parse_trec_document(file_path))
    spinner.succeed("Documents loaded.")

    spinner.start("Loading queries and qrels...")
    # Load queries and qrels
    wt_query_text_map = load_queries_file(depth_query_file)
    depth_qrels_by_query_id = load_qrels([depth_qrels_file])

    shallow_query_text_map_1 = load_queries(corpus_queries_path_1)
    shallow_query_text_map_2 = load_queries(corpus_queries_path_2)
    # Combine the two query text maps
    shallow_query_text_map = {
        **shallow_query_text_map_1,
        **shallow_query_text_map_2,
    }

    shallow_qrels_by_query_id = load_qrels(
        [shallow_qrel_file_1, shallow_qrel_file_2]
    )

    spinner.succeed("Queries and qrels loaded.")

    spinner.start("Creating depth-based training set...")
    # Create Depth-based training set
    depth_training_set = create_depth_training_set(
        depth_qrels_by_query_id, wt_query_text_map, seed=90
    )
    spinner.succeed("Training set created.")

    spinner.start("Creating shallow-based training set...")
    # Create Shallow-based training set
    shallow_training_set = create_shallow_training_set(
        shallow_qrels_by_query_id, shallow_query_text_map, 752, 1, seed=90
    )

    spinner.start("Converting to Huggingface dataset...")
    # Convert to Huggingface dataset
    depth_dataset = training_set_to_dataset(
        depth_training_set, wt_query_text_map, doc_text_map
    )
    shallow_dataset = training_set_to_dataset(
        shallow_training_set, shallow_query_text_map, doc_text_map
    )
    spinner.succeed("Dataset created.")

    spinner.start("Saving dataset...")
    # Save the dataset
    depth_dataset.save_to_disk("../data/longeval/bm25/90/depth_based")
    shallow_dataset.save_to_disk("../data/longeval/bm25/90/shallow_based")
    spinner.succeed("Dataset saved.")

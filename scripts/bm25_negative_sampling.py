"""Generate subsets of the training data for model training."""

import random
from collections import defaultdict

import ir_datasets
from datasets import Dataset
from halo import Halo
from pyserini.search.lucene import LuceneSearcher


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
            f"Could not find {num_queries} queries with {num_rels_per_query} "
            f"relevance judgments."
        )

    return selected_queries


def map_ids_to_texts(dataset):
    """Map query and document IDs to their respective texts.

    This function creates a mapping from query and document IDs to their
    respective texts using the provided dataset from ir_datasets.

    Args:
        dataset: The dataset to extract the texts from.

    Returns:
        dict: A dictionary mapping query IDs to their texts.
        dict: A dictionary mapping document IDs to their texts.
    """
    qid_to_text = {}
    docid_to_text = {}

    for query in dataset.queries_iter():
        qid_to_text[query.query_id] = query.text

    for doc in dataset.docs_iter():
        docid_to_text[doc.doc_id] = doc.text

    return qid_to_text, docid_to_text


def create_training_set(
    dataset,
    query_text_map,
    num_queries,
    num_rels_per_query,
    neg_sample_prop=1,
    seed=42,
    skip_top_k=50,
    combine_trec_datasets=False,
):
    """Generates a training set with BM25-based negative sampling.

    Args:
        dataset: An `ir_datasets` dataset object containing queries,
             documents, and relevance judgments.
        query_text_map (dict): A dictionary mapping query IDs to their texts.
        num_queries (int): The number of queries to select for the training set.
            The selection is based on the presence of a specified number of
            relevance judgments per query.
        num_rels_per_query (int): The number of relevant documents
            (positive examples) to include for each selected query.
            An equal number of negative examples will be selected using
            BM25-based negative sampling.
        seed (int, optional): A seed for the random number generator to ensure
            reproducibility of query selection and negative sampling.
            Defaults to 30.
        skip_top_k (int, optional): The number of top-ranked documents to skip
            when selecting negative examples. Defaults to 50.
        combine_trec_datasets (bool, optional): Whether to combine the TREC 2019
            and 2020 datasets. Used when a larger pool is needed.
            Defaults to False.

    Returns:
        list of tuples: A training set represented as a list of tuples,
            where each tuple contains a query ID, a document ID,
            and a binary label indicating relevance
            (1 for relevant, 0 for non-relevant).

    Raises:
        ValueError: If the `num_queries` or `num_rels_per_query` is less than 1.
    """
    random.seed(seed)
    training_set = []
    qrels_by_query_id = defaultdict(list)
    docid_set = set()

    for qrel in dataset.qrels_iter():
        qrels_by_query_id[qrel.query_id].append((qrel.doc_id, qrel.relevance))
        docid_set.add(qrel.doc_id)

    if combine_trec_datasets:
        for qrel in dataset_trec_2020.qrels_iter():
            qrels_by_query_id[qrel.query_id].append(
                (qrel.doc_id, qrel.relevance)
            )
            docid_set.add(qrel.doc_id)

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
        # hits = searcher.search(query_text, k=100 + len(positive_docs))
        hits = searcher.search(query_text, k=100 + len(positive_docs) * 4 * neg_sample_prop)
        for hit in hits[skip_top_k:]:
            if (
                hit.docid not in positive_docs
                and len(negative_docs) < num_rels_per_query * neg_sample_prop
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


def check_training_set(training_set):
    """Check the size and number of positive and negative examples.

    Args:
        training_set: The training set to check.
        Created with create_training_set.
    """
    # Total size of the training set
    total_size = len(training_set)

    # Count positive and negative examples
    num_positive = sum(1 for _, _, label in training_set if label == 1)
    num_negative = total_size - num_positive

    print(f"Total size of the training set: {total_size}")
    print(f"Number of positive examples: {num_positive}")
    print(f"Number of negative examples: {num_negative}")


if __name__ == "__main__":
    spinner = Halo(text="Loading datasets...", spinner="dots")

    # Load the datasets
    spinner.start()
    dataset_trec_2019 = ir_datasets.load(
        "msmarco-passage/trec-dl-2019/judged"
    )
    dataset_trec_2020 = ir_datasets.load(
        "msmarco-passage/trec-dl-2020/judged"
    )
    dataset_train = ir_datasets.load("msmarco-passage/train/judged")
    spinner.succeed("Datasets loaded.")

    # Map IDs to texts
    spinner.start("Mapping IDs to texts...")
    query_text_map_trec_2019, doc_text_map_trec_2019 = map_ids_to_texts(
        dataset_trec_2019
    )
    query_text_map_trec_2020, doc_text_map_trec_2020 = map_ids_to_texts(
        dataset_trec_2020
    )

    combined_query_text_map = {
        **query_text_map_trec_2019,
        **query_text_map_trec_2020,
    }
    combined_doc_text_map = {**doc_text_map_trec_2019, **doc_text_map_trec_2020}

    query_text_map_train, doc_text_map_train = map_ids_to_texts(dataset_train)
    spinner.succeed("IDs mapped to texts.")

    # Initialize BM25 searcher
    searcher = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")

    # Seed for reproducibility
    seed = 92

    # Generate training sets

    # Depth based training sets
    # spinner.start("Creating depth-based training sets...")
    # depth_based_1 = create_training_set(
    #     dataset_trec_2019,
    #     combined_query_text_map,
    #     50,
    #     50,
    #     neg_sample_prop=20,
    #     seed=seed,
    #     combine_trec_datasets=True,
    # )
    # spinner.succeed("Depth-based 1 training set created.")
    # check_training_set(depth_based_1)

    # spinner.start("Creating depth-based training sets...")
    # depth_based_2 = create_training_set(
    #     dataset_trec_2019,
    #     combined_query_text_map,
    #     70,
    #     30,
    #     neg_sample_prop=20,
    #     seed=seed,
    #     combine_trec_datasets=True,
    # )
    # spinner.succeed("Depth-based 2 training set created.")
    # check_training_set(depth_based_2)

    # spinner.start("Creating depth-based training sets...")
    # depth_based_3 = create_training_set(
    #     dataset_trec_2019,
    #     combined_query_text_map,
    #     70,
    #     30,
    #     neg_sample_prop=8,
    #     seed=seed,
    #     combine_trec_datasets=True,
    # )
    # spinner.succeed("Depth-based 3 training set created.")
    # check_training_set(depth_based_3)

    # spinner.start("Creating depth-based training sets...")
    # depth_based_4 = create_training_set(
    #     dataset_trec_2019,
    #     combined_query_text_map,
    #     70,
    #     30,
    #     neg_sample_prop=10,
    #     seed=seed,
    #     combine_trec_datasets=True,
    # )
    # spinner.succeed("Depth-based 4 training set created.")
    # check_training_set(depth_based_4)

    spinner.succeed("Depth-based training sets created.")

    # Shallow based training sets
    # spinner.start("Creating shallow-based training sets...")
    # shallow_based_1 = create_training_set(
    #     dataset_train,
    #     query_text_map_train,
    #     2100,
    #     1,
    #     neg_sample_prop=8,
    #     seed=seed,
    # )
    # spinner.succeed("Shallow-based 1 training set created.")
    # check_training_set(shallow_based_1)

    # spinner.start("Creating shallow-based training sets...")
    # shallow_based_2 = create_training_set(
    #     dataset_train,
    #     query_text_map_train,
    #     1000,
    #     1,
    #     neg_sample_prop=1,
    #     seed=seed,
    # )
    # spinner.succeed("Shallow-based 2 training set created.")
    # check_training_set(shallow_based_2)

    spinner.start("Creating shallow-based training sets...")
    shallow_based_3 = create_training_set(
        dataset_train,
        query_text_map_train,
        50,
        1,
        neg_sample_prop=1,
        seed=seed,
    )
    spinner.succeed("Shallow-based 3 training set created.")
    check_training_set(shallow_based_3)

    spinner.start("Creating shallow-based training sets...")
    shallow_based_4 = create_training_set(
        dataset_train,
        query_text_map_train,
        25,
        1,
        neg_sample_prop=1,
        seed=seed,
    )
    spinner.succeed("Shallow-based 4 training set created.")
    check_training_set(shallow_based_4)

    # spinner.succeed("Shallow-based training sets created.")

    # Shuffle the training sets
    random.seed(30)
    # random.shuffle(depth_based_1)
    # random.shuffle(depth_based_2)
    # random.shuffle(shallow_based_1)
    # random.shuffle(shallow_based_2)

    # random.shuffle(depth_based_3)
    # random.shuffle(depth_based_4)
    random.shuffle(shallow_based_3)
    random.shuffle(shallow_based_4)

    # Convert to Huggingface datasets
    # spinner.start("Convert to Huggingface datasets...")
    # hf_dataset_depth_1 = training_set_to_dataset(
    #     depth_based_1,
    #     combined_query_text_map,
    #     combined_doc_text_map,
    # )
    # hf_dataset_depth_2 = training_set_to_dataset(
    #     depth_based_2,
    #     combined_query_text_map,
    #     combined_doc_text_map,
    # )
    # hf_dataset_shallow_1 = training_set_to_dataset(
    #     shallow_based_1, query_text_map_train, doc_text_map_train
    # )
    # hf_dataset_shallow_2 = training_set_to_dataset(
    #     shallow_based_2, query_text_map_train, doc_text_map_train
    # )

    # hf_dataset_depth_3 = training_set_to_dataset(
    #     depth_based_3,
    #     combined_query_text_map,
    #     combined_doc_text_map,
    # )
    # hf_dataset_depth_4 = training_set_to_dataset(
    #     depth_based_4,
    #     combined_query_text_map,
    #     combined_doc_text_map,
    # )
    hf_dataset_shallow_3 = training_set_to_dataset(
        shallow_based_3, query_text_map_train, doc_text_map_train
    )
    hf_dataset_shallow_4 = training_set_to_dataset(
        shallow_based_4, query_text_map_train, doc_text_map_train
    )

    spinner.succeed("Huggingface datasets created.")

    # Save the datasets
    # spinner.start("Saving datasets...")
    # hf_dataset_depth_1.save_to_disk(
    #     f"../data/hf_datasets/msmarcov1/test/bm25/one_to_one/v2/{seed}/depth_based_50_50_1000"
    # )
    # hf_dataset_depth_2.save_to_disk(
    #     f"../data/hf_datasets/msmarcov1/test/bm25/one_to_one/v2/{seed}/depth_based_70_30_600"
    # )

    # hf_dataset_shallow_1.save_to_disk(
    #    f"../data/hf_datasets/msmarcov1/test/bm25/one_to_one/v2/{seed}/shallow_based_2100_1_8"
    # )

    # hf_dataset_shallow_2.save_to_disk(
    #     f"../data/hf_datasets/msmarcov1/test/bm25/one_to_one/v2/{seed}/shallow_based_1000_1_1"
    # )

    # hf_dataset_depth_3.save_to_disk(
    #     f"../data/hf_datasets/msmarcov1/test/bm25/one_to_one/v2/{seed}/depth_based_70_30_240"
    # )
    # hf_dataset_depth_4.save_to_disk(
    #     f"../data/hf_datasets/msmarcov1/test/bm25/one_to_one/v2/{seed}/depth_based_70_30_300"
    # )

    hf_dataset_shallow_3.save_to_disk(
        f"../data/hf_datasets/msmarcov1/test/bm25/one_to_one/v2/{seed}/shallow_based_50_1_1"
    )
    hf_dataset_shallow_4.save_to_disk(
        f"../data/hf_datasets/msmarcov1/test/bm25/one_to_one/v2/{seed}/shallow_based_25_1_1"
    )

    spinner.succeed("Huggingface datasets saved.")

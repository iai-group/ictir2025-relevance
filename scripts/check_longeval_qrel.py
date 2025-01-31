from collections import defaultdict


def load_qrels(qrels_file):
    """Load qrels from a text file."""
    qrels = defaultdict(list)
    with open(qrels_file, "r") as file:
        for line in file:
            query_id, _, doc_id, relevance = line.strip().split()
            qrels[query_id].append((doc_id, int(relevance)))
    return qrels


def count_relevant_qrels(qrels):
    relevant_qrel_count = defaultdict(int)

    for query_id, qrel_list in qrels.items():
        for _, relevance in qrel_list:
            if relevance >= 1:
                relevant_qrel_count[query_id] += 1

    return relevant_qrel_count


def print_relevant_judgment_stats(judgments_count, dataset_name):
    # Number of queries with at least one relevant judgment
    num_queries_with_relevant_qrels = len(judgments_count)
    print(
        f"{dataset_name} - Number of queries with relevant judgments: {num_queries_with_relevant_qrels}"
    )
    # Print the number of relevant judgments for each query
    for query_id, count in judgments_count.items():
        print(f"Query ID: {query_id} has {count} relevant judgments")


if __name__ == "__main__":
    # Load qrels file
    qrels_file = "../data/collections/longeval/publish/French/Qrels/train.txt"
    qrels = load_qrels(qrels_file)

    # Count relevant judgments
    relevant_judgments_count = count_relevant_qrels(qrels)

    # Print relevant judgment stats
    print_relevant_judgment_stats(relevant_judgments_count, "Longeval Qrels")

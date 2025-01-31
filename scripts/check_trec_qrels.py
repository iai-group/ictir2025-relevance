from collections import defaultdict

import ir_datasets


def count_relevant_judgments(dataset):
    relevant_query_judgments_count = defaultdict(int)

    for qrel in dataset.qrels_iter():
        if qrel.relevance >= 1:
            relevant_query_judgments_count[qrel.query_id] += 1

    return relevant_query_judgments_count


def print_relevant_judgment_stats(judgments_count, dataset_name):
    # Number of queries with at least one relevant judgment
    num_queries_with_relevant_judgments = len(judgments_count)
    print(
        f"{dataset_name} - Number of queries with relevant judgments: {num_queries_with_relevant_judgments}"
    )
    # Print the number of relevant judgments for each query
    for query_id, count in judgments_count.items():
        print(f"Query ID: {query_id} has {count} relevant judgments")


trec_2022 = ir_datasets.load("msmarco-passage-v2/trec-dl-2022/judged")
trec_2021_judged = ir_datasets.load("msmarco-passage-v2/trec-dl-2021/judged")

relevant_judgments_count_2022 = count_relevant_judgments(trec_2022)
relevant_judgments_count_2021_judged = count_relevant_judgments(
    trec_2021_judged
)

print_relevant_judgment_stats(relevant_judgments_count_2022, "TREC 2022")
print_relevant_judgment_stats(
    relevant_judgments_count_2021_judged, "TREC 2021 Judged"
)

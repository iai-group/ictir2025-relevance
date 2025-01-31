from collections import defaultdict

from datasets import load_from_disk

# Load datasets
dataset = load_from_disk("../data/longeval/bm25/42/depth_based")


# Function to analyze dataset
def analyze_dataset(dataset, name):
    # Check the number of examples
    print(f"Number of examples in {name}: {len(dataset)}")

    # Check the number of unique queries
    unique_queries = set(item["query_id"] for item in dataset)

    print(f"Number of unique queries in {name}: {len(unique_queries)}")

    # Analyzing the number of positive and negative examples per query
    query_examples = defaultdict(int)
    query_pos_neg_examples = defaultdict(lambda: {"positive": 0, "negative": 0})

    # Iterate through each example in the dataset
    for example in dataset:
        query_examples[example["query_id"]] += 1
        if example["relevance"] == 1:
            query_pos_neg_examples[example["query_id"]]["positive"] += 1
        else:
            query_pos_neg_examples[example["query_id"]]["negative"] += 1

    # Print detailed info per query
    for query_id, counts in query_examples.items():
        pos = query_pos_neg_examples[query_id]["positive"]
        neg = query_pos_neg_examples[query_id]["negative"]
        print(
            f"Query ID: {query_id} has {counts} examples ({pos} positive, {neg} negative)"
        )


# Analyze each dataset
analyze_dataset(dataset, "Depth test")

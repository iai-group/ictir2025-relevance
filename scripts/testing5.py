from collections import defaultdict
import ir_datasets
from tqdm import tqdm 


def trec_deep_learning_statistics(dataset_name):
    """Print statistics for the TREC Deep Learning dataset."""
    dataset = ir_datasets.load(dataset_name)
    num_queries = 0
    num_docs = 0
    num_qrels = 0
    num_pos_qrels = 0

    for query in dataset.queries_iter():
        num_queries += 1

    # for doc in tqdm(dataset.docs_iter()):
    #     num_docs += 1

    for qrel in dataset.qrels_iter():
        num_qrels += 1
        if qrel.relevance > 0:
            num_pos_qrels += 1

    statistics = {
        "num_queries": num_queries,
        # "num_docs": num_docs,
        "num_qrels": num_qrels,
        "num_pos_qrels": num_pos_qrels,
    }
    return statistics

def get_number_of_positive_qrels_by_queryid(dataset_name):
    dataset = ir_datasets.load(dataset_name)
    
    qrel_numbers_by_query_id = defaultdict(int)
    
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            qrel_numbers_by_query_id[qrel.query_id] += 1
        
    return qrel_numbers_by_query_id

import pandas as pd 

if __name__ == "__main__":

    # Depth based datasets statistics

    # dataset_name = "msmarco-passage/trec-dl-2019/judged"
    # statistics = trec_deep_learning_statistics(dataset_name)
    # print(f"Statistics for {dataset_name}:")
    # for key, value in statistics.items():
    #     print(f"{key}: {value}")

    # dataset_name = "msmarco-passage/trec-dl-2020/judged"
    # statistics = trec_deep_learning_statistics(dataset_name)
    # print(f"Statistics for {dataset_name}:")
    # for key, value in statistics.items():
    #     print(f"{key}: {value}")

    # dataset_name = "msmarco-passage-v2/trec-dl-2021/judged"
    # statistics = trec_deep_learning_statistics(dataset_name)
    # print(f"Statistics for {dataset_name}:")
    # for key, value in statistics.items():
    #     print(f"{key}: {value}")

    # dataset_name = "msmarco-passage-v2/trec-dl-2022/judged"
    # statistics = trec_deep_learning_statistics(dataset_name)
    # print(f"Statistics for {dataset_name}:")
    # for key, value in statistics.items():
    #     print(f"{key}: {value}")


    # # Shallow based datasets statistics
    # dataset_name = "msmarco-passage/train/judged"
    # statistics = trec_deep_learning_statistics(dataset_name)
    # print(f"Statistics for {dataset_name}:")
    # for key, value in statistics.items():
    #     print(f"{key}: {value}")

    # dataset_name = "msmarco-passage/dev/2"
    # statistics = trec_deep_learning_statistics(dataset_name)
    # print(f"Statistics for {dataset_name}:")
    # for key, value in statistics.items():
    #     print(f"{key}: {value}")

    # dataset_name = "msmarco-passage-v2/train"
    # statistics = trec_deep_learning_statistics(dataset_name)
    # print(f"Statistics for {dataset_name}:")
    # for key, value in statistics.items():
    #     print(f"{key}: {value}")

    # dataset_name = "msmarco-passage-v2/dev1"
    # statistics = trec_deep_learning_statistics(dataset_name)
    # print(f"Statistics for {dataset_name}:")
    # for key, value in statistics.items():
    #     print(f"{key}: {value}")

    # qrels_count2021 = get_number_of_positive_qrels_by_queryid("msmarco-passage-v2/trec-dl-2021/judged")
    qrels_count2022 = get_number_of_positive_qrels_by_queryid("msmarco-passage-v2/trec-dl-2022/judged")

    #print(qrels_count2021)

    #combined_counts = qrels_count2021 | qrels_count2022

    # total_number_of_qrels = sum(combined_counts.values())
    # average_qrels = total_number_of_qrels / len(combined_counts)
    # print(f"Total number of qrels: {total_number_of_qrels}")
    # print(f"Average number of qrels: {average_qrels}")

    # result = [key for key in qrels_count2021 if key in qrels_count2022 and qrels_count2021[key] > 40000 and qrels_count2022[key] > 40000]
    # print(result)

    # pd.DataFrame.from_dict(qrels_count2022, orient="index").to_csv("combined_counts.csv")

    # import matplotlib.pyplot as plt

    items = list(qrels_count2022.items())

    items = [item for item in items if item[1] < 40000]

    query_ids, num_qrels = zip(*items)

    query_ids = list(query_ids)
    num_qrels = list(num_qrels)

    # plt.figure(figsize=(12, 6))
    # plt.bar(query_ids, num_qrels)
    # plt.xticks(rotation=90)
    # plt.title('Query IDs vs Number of Positive Qrels TREC DL 2022')
    # plt.xlabel('Query ID')
    # plt.ylabel('Number of Positive Qrels')
    # plt.tight_layout()
    
    # plt.savefig('bar.png')

    import matplotlib.pyplot as plt


    plt.figure(figsize=(10, 6))
    plt.hist(num_qrels, bins=30, edgecolor='black')
    plt.title('Query IDs vs Number of Positive Qrels TREC DL 2022')
    plt.ylabel('Number of Query IDs')
    plt.xlabel('Number of Positive Qrels')
    plt.grid(True)
    plt.tight_layout()

    # Guarda el gráfico como imagen
    plt.savefig('histograma.png')

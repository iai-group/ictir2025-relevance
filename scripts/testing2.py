from collections import defaultdict

# queries_id = defaultdict(str)
# with open('../data/collections/longeval/train/English/Queries/train.tsv', 'r') as reader:
#     for line in reader:
#         line = line.strip("\n").split("\t")
#         queries_id[line[0]] = line[1]
# print(len(queries_id))

qrels_by_query_id = defaultdict(tuple)
with open('../data/collections/longeval/train/French/Qrels/train.txt', 'r') as reader:
    for line in reader:
        line = line.strip("\n").split(" ")
        if int(line[3]) > 0:
            qrels_by_query_id[line[0]] = (line[2], line[3])

statistics = defaultdict(int)
for query_id in qrels_by_query_id:
    if int(qrels_by_query_id[query_id][1]) > 0:
        statistics[query_id] += 1

# print(set(queries_id.keys()) == set(qrels_by_query_id.keys()))
print(len(statistics))
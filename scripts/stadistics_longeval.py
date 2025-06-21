from collections import defaultdict

qrels_by_query_ids = defaultdict(list)
with open('../data/collections/longeval/longeval-relevance-judgements/heldout-test.txt', 'r') as reader:
    for line in reader:
        line = line.strip("\n").split(" ")
        if int(line[3]) > 0:
            qrels_by_query_ids[line[0]].append((line[2], int(line[3])))

print(len(qrels_by_query_ids.keys()))

queries_ids = list(qrels_by_query_ids.keys())

stadistics_qrels = defaultdict(int)
for query in queries_ids:
    stadistics_qrels[query] = len(qrels_by_query_ids[query])

print(stadistics_qrels)

i = 0
j = 0
k = 0
l = 0
m = 0
for query in stadistics_qrels.keys():
    if stadistics_qrels[query] > 2:
        i += 1
    if stadistics_qrels[query] > 5:
        j += 1
    if stadistics_qrels[query] > 7:
        k += 1
    if stadistics_qrels[query] > 10:
        l += 1

print(i)
print(j)
print(k)
print(l)
from collections import defaultdict

qrels_by_query_ids1 = defaultdict(list)
with open('../data/collections/longeval/longeval-relevance-judgements/heldout-test.txt', 'r') as reader:
    for line in reader:
        line = line.strip("\n").split(" ")
        if int(line[3]) > 0:
            qrels_by_query_ids1[line[0]].append((line[2], int(line[3])))

qrels_by_query_ids2 = defaultdict(list)
with open('../data/collections/longeval/annotations_all.qrels', 'r') as reader:
    for line in reader:
        line = line.strip("\n").split(" ")
        if int(line[3]) > 0:
            qrels_by_query_ids2[line[0]].append((line[2], int(line[3])))

qrels_by_query_ids3 = defaultdict(list)
with open('../data/collections/longeval/annotations_wt.qrels', 'r') as reader:
    for line in reader:
        line = line.strip("\n").split(" ")
        if int(line[3]) > 0:
            qrels_by_query_ids3[line[0]].append((line[2], int(line[3])))

    
queries1 = set(qrels_by_query_ids1.keys())
queries2 = set(qrels_by_query_ids2.keys())
queries3 = set(qrels_by_query_ids3.keys())

print(len(queries1))
print(len(queries2))
print(len(queries3))

if not queries1.isdisjoint(queries2):
    print("They share elements.")
if not queries1.isdisjoint(queries3):
    print("They share elements.")
if not queries2.isdisjoint(queries3):
    print("They share elements.")

depth_queries = queries2.union(queries3)
print(len(depth_queries))

all_queries = queries1.union(queries2).union(queries3)
print(len(all_queries))


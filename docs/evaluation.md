# Evaluation

The evaluation process for the fine-tuned models involves using them as rerankers and calculating the performance metrics (MAP, NDCG, and MRR) using the `trec_eval` software.
You can download the `trec_eval` software from the [TREC Eval GitHub](https://github.com/usnistgov/trec_eval).

## Reranking File

The script for reranking, `reranking.py`, has been used for reranking in both the MS MARCO and LongEval experiments.
In the script, we have commented out lines based on which dataset we are working with.

## Indexing

For the training sets based on the MS MARCO dataset, we used the prebuilt index provided by Pyserini.
However, for the LongEval dataset, we had to create our own index.
Pyserini provides an easy method to do this.

### Creating an Index with Pyserini

1. **Install Pyserini:** Make sure you have Pyserini installed. If not, you can follow the instructions on [Pyserinis GitHub](https://github.com/castorini/pyserini?tab=readme-ov-file).
2. **Prepare The Data:** The LongEval dataset can be found [here](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5010). Make sure data is in JSON format or any format supported by Pyserini.
3. **Build the Index:** Invoke the Pyserini indexer:

   ```bash
   python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input tests/resources/sample_collection_jsonl \
    --index indexes/sample_collection_jsonl \
    --generator DefaultLuceneDocumentGenerator \
    --threads 9 \
    --storePositions --storeDocvectors --storeRaw
   ```

    - `--collection JsonCollection`: Specifies that the input data is in JSON format.
    - `--input /path/to/longeval/data`: The path to the directory containing your LongEval data.
    - `--index /path/to/output/index`: The path where the index will be stored.
    - `--generator DefaultLuceneDocumentGenerator`: The document generator to use.
    - `--threads 9`: The number of indexing threads to use.

Adjust the paths and number of threads as needed.

## Evaluate with TREC Eval

Use the `reranking.py` script to rerank the documents. Uncomment the relevant lines in the script based on whether you are working with the MS MARCO or LongEval dataset.
The reranking script will create a **run** file, a file containing the top 10 relevant documents for each query and their score.
Once you have your reranked results, use the `trec_eval` software to calculate the performance metrics. Run the following command:

```bash
trec_eval -m map -m ndcg -m recip_rank path_to_qrels path_to_run_file
```

## In case command is not working

```bash
export PATH=$PATH:/path/to/trec_eval
```
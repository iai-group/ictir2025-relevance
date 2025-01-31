def clean_qrel_file(input_qrel_path, output_qrel_path):
    """
    Remove duplicate lines from a QREL file.

    Parameters:
    - input_qrel_path: Path to the input QREL file.
    - output_qrel_path: Path to save the cleaned QREL file.
    """
    seen = set()
    with open(input_qrel_path, "r") as infile, open(
        output_qrel_path, "w"
    ) as outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) == 4:
                query_id, _, doc_id, relevance = parts
                key = (query_id, doc_id)
                if key not in seen:
                    outfile.write(line)
                    seen.add(key)


# Paths to your input and output QREL files
clean_qrel_file("annotations_st.qrels", "cleaned_annotations_st.qrels")
clean_qrel_file("annotations_lt.qrels", "cleaned_annotations_lt.qrels")

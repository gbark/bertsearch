"""
Example script to create elasticsearch documents.
"""
import argparse
import json

import pandas
from bert_serving.client import BertClient

bert_client = BertClient(output_fmt="list")


def create_document(doc, emb, index_name):
    return {
        "_op_type": "index",
        "_index": index_name,
        "text": doc["text"],
        "title": doc["title"],
        "text_vector": emb,
    }


def load_dataset(path):
    docs = []
    dataframe = pandas.read_csv(path)
    for row in dataframe.iterrows():
        series = row[1]
        doc = {"title": series.Name, "text": series.Temperament}
        docs.append(doc)
    return docs


def bulk_predict(docs, batch_size=256):
    """Predict bert embeddings."""
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        texts = [doc["text"] for doc in batch_docs]
        embeddings = bert_client.encode(texts)
        for embedding in embeddings:
            yield embedding


def main(args):
    docs = load_dataset(args.data)
    with open(args.save, "w") as f:
        for doc, embedding in zip(docs, bulk_predict(docs)):
            d = create_document(doc, embedding, args.index_name)
            f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creating elasticsearch documents.")
    parser.add_argument("--data", help="data for creating documents.")
    parser.add_argument("--save", default="documents.jsonl", help="created documents.")
    parser.add_argument(
        "--index_name", default="jobsearch", help="Elasticsearch index name."
    )
    args = parser.parse_args()
    main(args)

from elasticsearch import Elasticsearch
from tqdm import tqdm
from elasticsearch.helpers import streaming_bulk


def phrases_generator(path):
    with open(path) as f:
        for ix, line in enumerate(f):
            line = line.strip()
            doc = {"_id": ix, "phrase":line}
            yield doc


if __name__ == "__main__":

    with Elasticsearch() as es:

        phrases = phrases_generator('/home/enrique/Downloads/QASC_Corpus/QASC_Corpus.txt')
        progress = tqdm(unit="docs", desc="Indexing QASC corpus")
        successes = 0
        # for ok, action in streaming_bulk(es, index="qascfr-0001", actions=phrases):
        #     progress.update(1)
        #     successes += ok

        print(f"Indexed {successes} documents")

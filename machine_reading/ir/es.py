from elasticsearch import Elasticsearch
from functools import lru_cache
from typing import List, Tuple

special_characters = '+ - && || ! { } [ ] ^ " ~ * ? : \\ / . ,'.split()


def clean_query(text):
    for ch in special_characters:
        text = text.replace(ch, '')
    return text


class QASCIndexSearcher:

    def __init__(self):
        # initialize machine_reading
        self._client = Elasticsearch()

    # function to retrieve results based on a query string
    #@lru_cache(maxsize=2048)
    def search(self, query_string: str, max_hits: int) -> List[Tuple[str, float]]:
        query_string = clean_query(query_string)

        body = {
            "query": {
                "query_string": {
                    "query": query_string,
                    "default_field": "phrase"
                }
            },
            "size": max_hits
        }

        es = self._client

        res = es.search(body, index="qascfr-0001")

        ret = list()
        for elem in res['hits']['hits']:
            doc = elem['_source']['phrase']
            score = elem['_score']
            ret.append((doc, score))

        return ret


if __name__ == "__main__":
    # get queries and display results
    searcher = QASCIndexSearcher()
    running = True
    while running:
        query_string = input('> ')
        if query_string is None:
            running = False
        else:
            print(list(searcher.search(query_string, 10)))

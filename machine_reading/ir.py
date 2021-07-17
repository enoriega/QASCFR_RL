from functools import lru_cache
from typing import Tuple, List

import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.store import SimpleFSDirectory


def clean_query(text):
    special_characters = '+ - && || ! ( ) { } [ ] ^ " ~ * ? : \ / . ,'.split()
    for ch in special_characters:
        text = text.replace(ch, '')
    return text


class QASCIndexSearcher:

    def __init__(self, index_dir):
        self._index_dir = index_dir

        # initialize machine_reading
        lucene.initVM()
        directory = SimpleFSDirectory(Paths.get(self._index_dir))
        index_reader = DirectoryReader.open(directory)
        self._searcher = IndexSearcher(index_reader)
        analyzer = WhitespaceAnalyzer()
        self._query_parser = QueryParser('phrase', analyzer)  # This is the name of the field we search for

    # function to retrieve results based on a query string
    @lru_cache(maxsize=2048)
    def search(self, query_string: str, max_hits: int) -> List[Tuple[str, float]]:
        query_string = clean_query(query_string)
        query = self._query_parser.parse(query_string)
        top_docs = self._searcher.search(query, max_hits)
        ret = list()
        for score_doc in top_docs.scoreDocs:
            doc = self._searcher.doc(score_doc.doc)
            ret.append((doc.get('phrase'), score_doc.score))

        return ret


if __name__ == "__main__":
    # get queries and display results
    searcher = QASCIndexSearcher('data/lucene_index')
    running = True
    while running:
        query_string = input('> ')
        if query_string is None:
            running = False
        else:
            print(list(searcher.search(query_string, 10)))

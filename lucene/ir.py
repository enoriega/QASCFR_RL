from typing import Tuple, List

import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.store import SimpleFSDirectory


class QASCIndexSearcher:

    def __init__(self, index_dir, max_hits):
        self._index_dir = index_dir
        self._max_hits = max_hits

        # initialize lucene
        lucene.initVM()
        directory = SimpleFSDirectory(Paths.get(self._index_dir))
        index_reader = DirectoryReader.open(directory)
        self._searcher = IndexSearcher(index_reader)
        analyzer = WhitespaceAnalyzer()
        self._query_parser = QueryParser('phrase', analyzer)  # This is the name of the field we search for

    # function to retrieve results based on a query string
    def search(self, query_string: str) -> List[Tuple[str, float]]:
        query = self._query_parser.parse(query_string)
        top_docs = self._searcher.search(query, self._max_hits)
        ret = list()
        for score_doc in top_docs.scoreDocs:
            doc = self._searcher.doc(score_doc.doc)
            list.append((doc.get('phrase'), score_doc.score))

        return ret


if __name__ == "__main__":
    # get queries and display results
    searcher = QASCIndexSearcher('data/lucene_index', 10)
    running = True
    while running:
        query_string = input('> ')
        if query_string is None:
            running = False
        else:
            print(list(searcher.search(query_string)))

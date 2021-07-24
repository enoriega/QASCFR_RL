import math
from collections import Counter
from typing import Sequence, Iterable


class FrequencyCounter:
    def __init__(self, docs: Iterable[str]) -> None:
        self.num_docs = len(docs)
        self.term_freqs = Counter()
        self.doc_freqs = Counter()
        self.idf = dict()

        for doc in docs:
            seen = set()
            for term in doc:
                self.term_freqs[term] += 1
                if term not in seen:
                    self.doc_freqs[term] += 1
                    seen.add(term)

        self.idf = {k: math.log(self.num_docs / v) for k, v in self.doc_freqs.items()}
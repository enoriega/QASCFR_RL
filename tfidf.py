""" Script to generate the TF-IDF over the corpus """
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Union, Sequence

import numpy as np
import spacy
from tqdm import tqdm

from nlp import preprocess


class TfIdfHelper:
    """ Helper class to compute and fetch tfidf scores 'globally' in the corpus (i.e. not document wise term counts)"""
    def __init__(self, vocabulary:Sequence[str], tfidf: np.ndarray) -> None:

        self._voc = vocabulary
        self._voc_ix = {t: ix for ix, t in enumerate(vocabulary)}
        self._vector = tfidf

    def __getitem__(self, item:Union[str, Sequence[str]]) -> np.ndarray:
        """ Returns the global tfidf score for each requested element """

        if type(item) == str:
            item = [item]

        indices = [self._voc_ix[i] for i in item if i in self._voc_ix]
        return self._vector[indices]

    @classmethod
    def build_tfidf(cls, corpus:Union[str, Path]):
        """ Builds the tfidf scores for the corpus """

        if type(corpus) == str:
            corpus = Path(corpus)

        # Check to see if a cached version of the numbers exists already
        cached = corpus.with_suffix(".pickle")

        if cached.exists():
            with cached.open("rb") as f:
                data = pickle.load(f)
        else:
            nlp = spacy.blank("en")

            num_pattern = re.compile(r"[+\-.]?\d+([.,/:]\d+)?[enf]?")

            with corpus.open('r') as f:
                term_frequencies = Counter()
                document_frequencies = Counter()
                num_docs = 0
                doc_terms = set()
                doc_sentences = list()
                for line in tqdm(f):
                    if line == '\n':
                        # Change the doc
                        # Ignore the first line, because it is the hash of the doc
                        for sentence in preprocess(tuple(doc_sentences[1:]), nlp):
                            for token in sentence:
                                token = token.strip()
                                if token:
                                    # Replace numbers
                                    if num_pattern.match(token):
                                        # token = "NUM"
                                        continue  # Comment here to count the numbers
                                    term_frequencies[token] += 1
                                    doc_terms.add(token)
                        for term in doc_terms:
                            document_frequencies[term] += 1

                        if len(doc_terms) > 0:
                            num_docs += 1

                        doc_terms = set()
                        doc_sentences = list()
                    else:
                        doc_sentences.append(line)

            # Sort the keys and make them our vocabulary
            sorted_keys = list(sorted(document_frequencies))

            # Make the counters be numpy arrays
            term_frequencies = np.asarray([term_frequencies[k] for k in sorted_keys])
            document_frequencies = np.asarray([document_frequencies[k] for k in sorted_keys])

            # This is the corpus tf-idf scores. No need for smoothing as it's not document level
            tfidf = term_frequencies * np.log2(num_docs/document_frequencies)

            data = cls(sorted_keys, tfidf)

            with cached.open("wb") as f:
                pickle.dump(data, f)

        return data

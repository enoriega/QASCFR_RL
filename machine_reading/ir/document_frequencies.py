import math
import pickle
from pathlib import Path
from typing import Dict, Sequence, Mapping

import spacy
from tqdm import tqdm

import utils
from nlp import preprocess
import itertools as it
from collections import Counter


class FrequencyCounter:
    def __init__(self, docs: Sequence[str]) -> None:
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


def compute_frequencies(path: Path) -> FrequencyCounter:
    pipeline = spacy.load("en_core_web_sm")
    with path.open('r') as f:
        processed = preprocess(tqdm(f, desc="Pre-processing corpus"), pipeline)
        counter = FrequencyCounter(tqdm(processed, desc="Counting terms"))
    return counter


if __name__ == "__main__":
    # Read the config values
    config = utils.read_config()
    corpus_path = Path(config['files']['corpus_path'])
    frequencies_path = Path(config['files']['frequencies_path'])
    frequencies = compute_frequencies(corpus_path)

    with frequencies_path.open('wb') as f:
        pickle.dump(frequencies, f)

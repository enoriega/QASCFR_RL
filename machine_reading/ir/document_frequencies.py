import pickle
from functools import partial
from pathlib import Path

import spacy
from tqdm import tqdm

import utils
from machine_reading.ir.frequency_counter import FrequencyCounter
from nlp import preprocess
import itertools as it

from multiprocessing import Pool


def partial_process(pipeline, elements):
    return preprocess(elements, pipeline)


def compute_frequencies(path: Path) -> FrequencyCounter:
    pipeline = spacy.load("en_core_web_sm")
    with path.open('r') as f:
        elems = f
        g = partial(partial_process, pipeline)
        with Pool(16) as ctx:
            processed = ctx.map(g, elems)
            # processed = preprocess(tqdm(elems, desc="Pre-processing corpus"), pipeline)
        counter = FrequencyCounter(tqdm(list(it.chain.from_iterable(processed)), desc="Counting terms"))
    return counter


if __name__ == "__main__":
    # Read the config values
    config = utils.read_config()
    corpus_path = Path(config['files']['corpus_path'])
    frequencies_path = Path(config['files']['frequencies_path'])
    frequencies = compute_frequencies(corpus_path)

    with frequencies_path.open('wb') as f:
        pickle.dump(frequencies, f)

""" This script implements the two-step IR baseline methods, described at https://arxiv.org/abs/1910.11473 """
import itertools
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Sequence, Tuple, Mapping, List, Any
from pathlib import Path

import pandas as pd
import spacy
from nltk import SnowballStemmer
from pandas import DataFrame
from spacy import Language
import itertools as it
import re

from tqdm import tqdm

import utils
from machine_reading.ir.es import QASCIndexSearcher
from machine_reading.ir.es_search import EsSearch
from nlp import preprocess
from parsing import read_problems, QASCItem


def retrieve_candidates(item: QASCItem):
    searcher = EsSearch(indices='ai2_qasc', twostep_qa=True, must_match_qa=True)

    question = item.question
    choice = item.answer

    return list(searcher.get_hits_for_question(question, [choice]).values())[0]



def two_step_retrieval(path: Path) -> Mapping[QASCItem, Any]:
    """ Run the two step retrieval process over the elements in path """

    # Parse the dataset file
    items = read_problems(path)

    language = spacy.load("en_core_web_sm")

    # Retrieve the candidate documents for each element
    results = dict()





    # # Single process
    # for item in tqdm(items, desc='Retrieving matches'):
    #     candidates = retrieve_candidates(item)
    #     results[item] = candidates

    # Multi processing
    progress = tqdm(desc='Retrieving matches', total=len(items))
    with ProcessPoolExecutor() as executor:

        futures = dict()
        for item in items:
            future = executor.submit(retrieve_candidates, item)
            futures[future] = item

        for future in as_completed(futures.keys()):
            item = futures[future]
            candidates = future.result()
            results[item] = candidates
            progress.update(1)

    # Return the map to the retrieved results
    return results

def find_index(txt, candidates):

    for ix, (pair, _) in enumerate(candidates):
        if txt in pair:
            return ix

    return None


# def eval(data: Mapping[QASCItem, RetrievalResults]) -> DataFrame:
#     """ Generates data frame with the analysis of the results """
#
#     rows = list()
#
#     for item, results in data.items():
#         f1, f2 = item.gt_path
#         f1_ix = find_index(f1, results)
#         f2_ix = find_index(f2, results)
#
#         rows.append({
#             'q': item.question,
#             'a': item.answer,
#             'f1': f1,
#             'f1_ix': f1_ix,
#             'f2': f2,
#             'f2_ix': f2_ix
#         })
#
#     frame = pd.DataFrame(rows)
#
#     return frame

def main(path: Path) -> None:
    retrieval_data = two_step_retrieval(path)
    with open('retrieval_results_ai2.pickle', 'wb') as f:
        pickle.dump(retrieval_data, f)

    # with open('retrieval_results2.pickle', 'rb') as f:
    #     retrieval_data = pickle.load(f)

    # frame = eval(retrieval_data)
    # frame.to_pickle("heuristic_baseline.pickle")
    # print(frame.head())

if __name__ == "__main__":
    config = utils.read_config()
    main(Path(config['files']['train_file']))



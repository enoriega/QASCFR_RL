""" This script implements the two-step IR baseline methods, described at https://arxiv.org/abs/1910.11473 """
import itertools
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Sequence, Tuple, Mapping
from pathlib import Path

import pandas as pd
import spacy
from nltk import SnowballStemmer
from pandas import DataFrame
from spacy import Language
import itertools as it

from tqdm import tqdm

import utils
from machine_reading.ir.es import QASCIndexSearcher
from nlp import preprocess
from parsing import read_problems, QASCItem

RetrievalResults = Sequence[Tuple[Tuple[str, str], float]]


def qualifies(phrase: str) -> bool:
    """ Filtering criteria for the heuristic, according to the paper """
    return True # TODO implement this correctly


def retrieve_candidates(item: QASCItem, language: Language) -> RetrievalResults :
    searcher = QASCIndexSearcher()

    q_terms = set(preprocess(item.question, language, stem=False))
    a_terms = set(preprocess(item.answer, language, stem=False))

    q_a = set(it.chain(q_terms, a_terms))
    stemmer = SnowballStemmer(language='english')
    q_a_stem = set(stemmer.stem(t) for t in q_a)

    f_1 = searcher.search(' '.join(q_a), 20)

    pairs = list()
    for phrase, p_score in f_1:
        if qualifies(phrase):
            phrase_terms = set(preprocess(phrase, language, stem=False))
            phrase_terms_stem = {stemmer.stem(t) for t in phrase_terms}
            left = q_terms - phrase_terms
            right =  phrase_terms - q_terms
            if ' '.join(left).strip():
                query = f"({' OR '.join(left)})"
                if len(right) > 0:
                    query += f" AND ({' OR '.join(right)})"
            else:
                if len(right) > 0:
                    query = f"({' OR '.join(right)})"
                else:
                    continue

            try:
                selected = 0
                x = searcher.search(query, 500)
                for candidate, c_score in x:
                    candidate_terms = set(preprocess(candidate, language, stem=False))
                    candidate_terms_stem = {stemmer.stem(t) for t in candidate_terms}
                    # Filter any candidate pair that don't share terms with q+a
                    if phrase != candidate and\
                            len((phrase_terms_stem - q_a_stem) & candidate_terms_stem) > 0 and\
                            len((q_a_stem - phrase_terms_stem) & candidate_terms_stem) > 0 and\
                            len(q_a_stem & (phrase_terms_stem | candidate_terms_stem)) > 0:
                        pairs.append(((phrase, candidate), p_score + c_score))
                        selected += 1
                    if selected >= 4:
                        break
            except:
                print("error for query " + query)

            x = 0

    return sorted(pairs, key=lambda x: x[1], reverse=True)


def two_step_retrieval(path: Path) -> Mapping[QASCItem, RetrievalResults]:
    """ Run the two step retrieval process over the elements in path """

    # Parse the dataset file
    items = read_problems(path)

    language = spacy.load("en_core_web_sm")

    # Retrieve the candidate documents for each element
    results = dict()



    # # Single process
    # for item in tqdm(items, desc='Retrieving matches'):
    #     candidates = retrieve_candidates(item, language)
    #     results[item] = candidates

    # Multi processing
    progress = tqdm(desc='Retrieving matches', total=len(items))
    with ProcessPoolExecutor() as executor:

        futures = dict()
        for item in items:
            future = executor.submit(retrieve_candidates, item, language)
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


def eval(data: Mapping[QASCItem, RetrievalResults]) -> DataFrame:
    """ Generates data frame with the analysis of the results """

    rows = list()

    for item, results in data.items():
        f1, f2 = item.gt_path
        f1_ix = find_index(f1, results)
        f2_ix = find_index(f2, results)

        rows.append({
            'q': item.question,
            'a': item.answer,
            'f1': f1,
            'f1_ix': f1_ix,
            'f2': f2,
            'f2_ix': f2_ix
        })

    frame = pd.DataFrame(rows)

    return frame

def main(path: Path) -> None:
    retrieval_data = two_step_retrieval(path)
    with open('retrieval_results2.pickle', 'wb') as f:
        pickle.dump(retrieval_data, f)

    # with open('retrieval_results.pickle', 'rb') as f:
    #     retrieval_data = pickle.load(f)
    #
    # frame = eval(retrieval_data)
    # # frame.to_pickle("heuristic_baseline2.pickle")
    # print(frame.head())

if __name__ == "__main__":
    config = utils.read_config()
    main(Path(config['files']['train_file']))



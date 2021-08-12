import csv
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from typing import cast

import pandas as pd
import spacy
from gensim.models import KeyedVectors
from tqdm import tqdm
import itertools as it

import utils
from baselines.cascade import CascadeAgent
from machine_reading.ie import RedisWrapper
# from nlp import EmbeddingSpaceHelper
from parsing import read_problems, QASCItem
from environment import QASCInstanceEnvironment
from machine_reading.ir.es import QASCIndexSearcher
from utils import build_rng

index = None
redis = None
embeddings = None
language = None


def contains_gt(instance: QASCItem, paths):
    facts = instance.gt_path
    total = False
    partial = False
    for p in paths:
        nodes = set(p)
        if facts[0] in nodes and facts[1] in nodes:
            total = True
            partial = True
            break
        elif facts[0] in nodes or facts[1] in nodes:
            partial = True
    return total, partial


def crunch_numbers(results, output_main, output_paths):
    main_rows = list()
    aux_rows = list()
    for (instance, seed), (env, paths) in results.items():
        main_row, aux = make_csv_row(env, instance, paths, seed)
        main_rows.append(main_row)
        aux_rows.extend(aux)
    frame = pd.DataFrame(main_rows)
    paths_frame = pd.DataFrame(aux_rows)

    # writer = pd.ExcelWriter('qasc_baseline_results.xlsx', engine='xlsxwriter')
    frame.to_csv(output_main)
    paths_frame.to_csv(output_paths)


def make_csv_row(env, instance, paths, seed):
    """ Prepares the data from a trail into rows to write to a csv """

    q, a = instance.question, instance.answer
    successful, partially_successful = contains_gt(instance, [paths])
    num_docs = env.num_docs
    num_paths = len(paths)
    iterations = env.iterations
    coverage = env.fr_score
    main_row = \
        {'q': q, 'a': a, 'seed': seed,
         'iterations': iterations,
         'docs': num_docs,
         'coverage': coverage,
         'success': successful,
         'partial_success': partially_successful,
         'paths': num_paths}

    aux_rows = list()

    if len(paths) > 0 and type(paths[0]) == str:
        paths = [paths]

    for path in paths:
        path_len = len(path)
        path_str = ' || '.join(path)
        aux_rows.append({'q': q, 'a': a, 'seed': seed, 'hops': path_len - 1, 'intermediate': path_str})

    return main_row, aux_rows


def test(instance, seed_state, ix):
    global index, redis, embeddings, language
    x = 1
    return 0.0


def schedule(instance, doc_universe, seed_state, ix):
    global index, redis, embeddings, language

    agent = CascadeAgent(seed_state)

    for seed in seed_state.randint(0, 100000, 1):
        try:
            # Instantiate the environment
            env = QASCInstanceEnvironment(instance, 10, True, 15, seed, index, redis, embeddings, language, doc_universe)
            result, outcome = agent.run(env)
            main_row, aux_rows = make_csv_row(env, instance, result, seed)
            return main_row, aux_rows
        except Exception as ex:
            print(f'Problem with instance {ix}')
            print(ex)


def main():
    # Read the config values

    global index, redis, embeddings, language

    config = utils.read_config()
    files_config = config['files']
    local_config = config['run_baseline']
    output_main = local_config['output_main']
    output_paths = local_config['output_paths']

    with open(files_config['retrieval_results'], 'rb') as f:
        data = pickle.load(f)
        doc_universes = dict()
        for item, results in data.items():
            docs = set(it.chain.from_iterable(v[0] for v in results))
            doc_universes[item] = docs

    instances = read_problems(files_config['train_file'])
    seed_state = build_rng(0)

    global index, redis, embeddings, language

    index = QASCIndexSearcher()
    redis = RedisWrapper()
    embeddings = cast(KeyedVectors, KeyedVectors.load('data/glove.840B.300d.kv'))
    language = spacy.load("en_core_web_sm")

    # results = dict()
    with open(output_main, 'w') as a, open(output_paths, 'w') as b:
        results_writer = csv.DictWriter(a, fieldnames=['q', 'a', 'seed', 'iterations', 'docs', 'success', 'partial_success', 'coverage', 'paths'])
        paths_writer = csv.DictWriter(b, fieldnames=['q', 'a', 'seed', 'hops', 'intermediate'])

        results_writer.writeheader()
        paths_writer.writeheader()

        # This is necessary to share the global variables among processes, but only works on unix-like platforms
        import multiprocessing as mp
        mp.set_start_method('fork')

        # Multi processing
        with ThreadPoolExecutor(max_workers=11) as ctx:
            futures = list()
            progress = tqdm(desc="Running baseline over dataset", total=len(instances))
            for ix, instance in enumerate(instances):
                future = ctx.submit(schedule, instance, doc_universes[instance], seed_state, ix)
                futures.append(future)

            for future in as_completed(futures):
                data = future.result()
                if data:
                    main_row, aux_row = data
                    results_writer.writerow(main_row)
                    paths_writer.writerows(aux_row)
                progress.update(1)

        # # Single process
        # progress = tqdm(desc="Running baseline over dataset", total=len(instances))
        # for ix, instance in enumerate(instances):
        #     data = schedule(instance, doc_universes[instance], seed_state, ix)
        #     if data:
        #         main_row, aux_row = data
        #         results_writer.writerow(main_row)
        #         paths_writer.writerows(aux_row)
        #     progress.update(1)


    # crunch_numbers(results)


if __name__ == "__main__":
    main()

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import cast

import pandas as pd
import spacy
from gensim.models import KeyedVectors
from tqdm import tqdm

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
    ret = False
    for p in paths:
        nodes = set(p)
        if facts[0] in nodes and facts[1] in nodes:
            ret = True
            break
    return ret


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
    successful = contains_gt(instance, paths)
    num_docs = env.num_docs
    num_paths = len(paths)
    iterations = env.iterations
    main_row = \
        {'q': q, 'a': a, 'seed': seed,
         'iterations': iterations,
         'docs': num_docs,
         'success': successful,
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


def schedule(instance, seed_state, ix):
    global index, redis, embeddings, language

    agent = CascadeAgent(seed_state)

    for seed in seed_state.randint(0, 100000, 1):
        try:
            # Instantiate the environment
            env = QASCInstanceEnvironment(instance, 10, True, 15, seed, index, redis, embeddings, language)
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

    instances = read_problems(files_config['train_file'])
    seed_state = build_rng(0)

    global index, redis, embeddings, language

    index = QASCIndexSearcher()
    redis = RedisWrapper()
    embeddings = cast(KeyedVectors, KeyedVectors.load('data/glove.840B.300d.kv'))
    language = spacy.load("en_core_web_sm")

    # results = dict()
    with open(output_main, 'w') as a, open(output_paths, 'w') as b:
        results_writer = csv.DictWriter(a, fieldnames=['q', 'a', 'seed', 'iterations', 'docs', 'success', 'paths'])
        paths_writer = csv.DictWriter(b, fieldnames=['q', 'a', 'seed', 'hops', 'intermediate'])

        results_writer.writeheader()
        paths_writer.writeheader()

        # This is necessary to share the global variables among processes, but only works on unix-like platforms
        import multiprocessing as mp
        mp.set_start_method('fork')

        with ProcessPoolExecutor(max_workers=12) as ctx:
            futures = list()
            progress = tqdm(desc="Running baseline over dataset", total=len(instances))
            for ix, instance in enumerate(instances):
                future = ctx.submit(schedule, instance, seed_state, ix)
                futures.append(future)

            for future in as_completed(futures):
                data = future.result()
                if data:
                    main_row, aux_row = data
                    results_writer.writerow(main_row)
                    paths_writer.writerows(aux_row)
                progress.update(1)


    # crunch_numbers(results)


if __name__ == "__main__":
    main()

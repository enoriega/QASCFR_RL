import csv

import pandas as pd
from tqdm import tqdm

import utils
from baselines.cascade import CascadeAgent
from machine_reading.ie import RedisWrapper
from parsing import read_problems, QASCInstance
from environment import Environment
from machine_reading.ir import QASCIndexSearcher
from utils import build_rng


def contains_gt(instance: QASCInstance, paths):
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

    for path in paths:
        path_len = len(path)
        path_str = ' || '.join(path)
        aux_rows.append({'q': q, 'a': a, 'seed': seed, 'hops': path_len - 1, 'intermediate': path_str[1:][:-1]})

    return main_row, aux_rows


def main():
    # Read the config values
    config = utils.read_config()
    files_config = config['files']
    local_config = config['run_baseline']
    output_main = local_config['output_main']
    output_paths = local_config['output_paths']

    instances = read_problems(files_config['train_file'])
    seed_state = build_rng(0)
    agent = CascadeAgent(seed_state)
    lucene = QASCIndexSearcher(files_config['lucene_index_dir'])
    redis = RedisWrapper()

    # results = dict()
    with open(output_main, 'w') as a, open(output_paths, 'w') as b:
        results_writer = csv.DictWriter(a, fieldnames=['q', 'a', 'seed', 'iterations', 'docs', 'success', 'paths'])
        paths_writer = csv.DictWriter(b, fieldnames=['q', 'a', 'seed', 'hops', 'intermediate'])

        results_writer.writeheader()
        paths_writer.writeheader()

        for ix, instance in tqdm(enumerate(instances), desc="Running baseline over dataset", total=len(instances)):
            for seed in seed_state.randint(0, 100000, 15):
                try:
                    # Instantiate the environment
                    env = Environment(instance, 10, True, 15, seed, lucene, redis)
                    result = agent.run(env)
                    # results[(instance, seed)] = (env, result)
                    main_row, aux_rows = make_csv_row(env, instance, result, seed)
                    results_writer.writerow(main_row)
                    paths_writer.writerows(aux_rows)
                except Exception as ex:
                    print(f'Problem with instance {ix}')
                    print(ex)

    # crunch_numbers(results)


if __name__ == "__main__":
    main()

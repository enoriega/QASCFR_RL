import pandas as pd
from tqdm import tqdm

from baselines.cascade import CascadeAgent
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


def crunch_numbers(results):

    main_rows = list()
    aux_rows = list()
    for (instance, seed), (env, paths) in results.items():
        q, a = instance.question, instance.answer
        successful = contains_gt(instance, paths)
        num_docs = env.num_docs
        num_paths = len(paths)
        iterations = env.iterations
        main_rows.append(
            {'q': q, 'a': a, 'seed':seed,
             'iterations': iterations,
             'docs': num_docs,
             'success': successful,
             'paths': num_paths})
        for path in paths:
            path_len = len(path)
            path_str = ' || '.join(path)
            aux_rows.append({'q': q, 'a': a, 'seed':seed, 'hops': path_len-1, 'intermediate': path_str[1:][:-1]})
    frame = pd.DataFrame(main_rows)
    paths_frame = pd.DataFrame(aux_rows)

    writer = pd.ExcelWriter('qasc_baseline_results.xlsx', engine='xlsxwriter')
    frame.to_excel(writer, sheet_name='main')
    paths_frame.to_excel(writer, sheet_name='paths')
    writer.close()


def main():
    # TODO parameterize paths
    instances = read_problems("/home/enrique/Downloads/QASC_Dataset/train.jsonl")
    seed_state = build_rng(0)
    agent = CascadeAgent(seed_state)
    lucene = QASCIndexSearcher('data/lucene_index')

    results = dict()
    for ix, instance in tqdm(enumerate(instances), desc="Running baseline over dataset"):
        for seed in seed_state.randint(0, 100000, 15):
            try:
                # Instantiate the environment
                env = Environment(instance, 10, True, 15, seed, lucene)
                result = agent.run(env)
                results[(instance, seed)] = (env, result)
            except Exception as ex:
                print(f'Problem with instance {ix}')

    crunch_numbers(results)


if __name__ == "__main__":
    main()

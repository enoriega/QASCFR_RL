import itertools as it
import shelve
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Tuple, NamedTuple, Union, AbstractSet, Optional

from configobj import ConfigObj
from numpy.random import RandomState
from pandas import DataFrame
from rlpyt.utils.launching.affinity import encode_affinity
from tqdm import tqdm

from parsing import Results, Pair, QASCItem


# from environment import ResultsKey

class ResultsKey(NamedTuple):
    problem: QASCItem
    seed: int


def build_rng(global_seed: int) -> RandomState:
    """ Builds the numpy random state for reproducibility """
    rng = RandomState(global_seed)
    return rng


def build_results_frame(results: Mapping[ResultsKey, Results]) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """ Creates a data frame for analysis using the collected results """

    def build_outcomes_row(key, result):
        """ Helper function to avoid caching the elements in a list to construct the data frame """
        prob = key.problem
        a = prob.question
        b = prob.answer
        seed = key.seed
        outcome = result.outcome
        papers_read = result.phrases_read
        num_iterations = result.num_queries

        return [a, b, seed, outcome, papers_read, num_iterations]

    def build_queries_rows(key, result: Results):
        """ Helper function to avoid caching the elements in a list to construct the data frame """
        prob = key.problem
        a = prob.question
        b = prob.answer
        seed = key.seed

        queries = result.queries
        rows = list()
        for qix, (q, d) in enumerate(zip(queries, result.differential_phrases), start=1):
            rows.append([a, b, seed,  # The "Foreign Key"
                         qix,  # Index of the query
                         q.endpoints[0] if type(q.endpoints) == tuple else q.endpoints,  # Endpoint A
                         q.endpoints[1] if len(q.endpoints) == 2 else None,  # Endpoint B (only for AND or OR)
                         q.type,  # Query type
                         len(d)])  # Number of new docs added by the query

        return rows

    def build_paths_rows(key, result: Results):
        """ Helper function to avoid caching the elements in a list to construct the data frame """
        prob = key.problem
        a = prob.question
        b = prob.answer
        seed = key.seed

        path = result.connecting_paths

        if path is None:
            path = list()

        rows = list()
        for ix, e in enumerate(path, start=1):
            rows.append([a, b, seed,  # The "Foreign Key"
                         ix,  # Index of the entity
                         e])  # Identity of the entity on the winning path

        return rows

    # Build the outcomes frame
    outcome_frame = \
        DataFrame(
            (build_outcomes_row(k, r) for k, r in tqdm(results.items(), desc="Building outcomes frame")),
            columns=['entity_a', 'entity_b', 'seed', 'successful', 'papers', 'iterations']
        )

    outcome_frame.entity_a = outcome_frame.entity_a.astype('category')
    outcome_frame.entity_b = outcome_frame.entity_b.astype('category')

    # Build the queries frame
    queries_frame = \
        DataFrame(
            it.chain.from_iterable(build_queries_rows(k, r)
                                   for k, r in tqdm(results.items(), desc="Building queries frame")),
            columns=['entity_a', 'entity_b', 'seed', 'query_ix', 'endpoint_a', 'endpoint_b', 'kind', 'new_docs']
        )

    queries_frame.entity_a = queries_frame.entity_a.astype('category')
    queries_frame.entity_b = queries_frame.entity_b.astype('category')
    queries_frame.endpoint_a = queries_frame.endpoint_a.astype('category')
    queries_frame.endpoint_b = queries_frame.endpoint_b.astype('category')
    queries_frame.kind = queries_frame.kind.astype('category')

    # Build the found paths frame
    paths_frame = \
        DataFrame(
            it.chain.from_iterable(build_paths_rows(k, r)
                                   for k, r in tqdm(results.items(), desc="Building paths frame")),
            columns=['entity_a', 'entity_b', 'seed', 'index', 'entity']
        )

    paths_frame.entity_a = paths_frame.entity_a.astype('category')
    paths_frame.entity_b = paths_frame.entity_b.astype('category')
    paths_frame.entity = paths_frame.entity.astype('category')

    return outcome_frame, queries_frame, paths_frame


def build_indices(p: Union[str, Path]) -> \
        Tuple[Mapping[str, AbstractSet[Union[str, Pair]]], Mapping[Union[Pair, str], AbstractSet[str]]]:
    with shelve.open(str(p)) as db:
        paths = dict(db)

    pairs = it.chain.from_iterable(paths[k] for k in paths)

    inverted = defaultdict(set)
    index = defaultdict(set)
    for elem in pairs:
        for (src, dst), sources in zip(elem[0], elem[1]):
            key = Pair(src, dst)
            inverted[key] |= sources
            inverted[src] |= sources
            inverted[dst] |= sources
            for doc in sources:
                index[doc] |= {src, dst, key}

    return dict(index), dict(inverted)


# Fetched from https://docs.python.org/3.7/library/itertools.html
def partition(pred, iterable):
    """Use a predicate to partition entries into false entries and true entries"""
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    t1, t2 = it.tee(iterable)
    return it.filterfalse(pred, t1), filter(pred, t2)


def read_results_shelf(shelf_path: Path) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """ Reads the data frames from a shelf file """
    with shelve.open(str(shelf_path)) as store:
        outcomes_frame: DataFrame = store['outcomes']
        queries_frame: DataFrame = store['queries']
        paths_frame: DataFrame = store['paths']
    return outcomes_frame, queries_frame, paths_frame


def read_config(path: Optional[Path] = None) -> ConfigObj:
    """ Reads the specified configuration file. If no file is specified, uses the default file config.ini """

    if path is None:
        path = Path("config.ini")

    with path.open() as f:
        config = ConfigObj(f)

    return config


def get_affinity(config: ConfigObj, run_slot: Optional[int] = None) -> str:
    """ Builds an slot-affinity code from the configuration """

    train = config['rl_train']
    num_cpus = int(train['num_cpus'])
    cpus_per_run = int(train['cpus_per_run'])
    num_gpus = int(train['num_gpus'])
    gpus_per_run = int(train['gpus_per_run'])

    args = {
        "n_cpu_core": num_cpus,
        "cpu_per_run": cpus_per_run,
    }

    if num_gpus > 0:
        args["n_gpu"] = num_gpus
        args["gpu_per_run"] = gpus_per_run

    if not run_slot:
        run_slot = 0

    slot_affinity_code = encode_affinity(run_slot=run_slot, **args)

    return slot_affinity_code
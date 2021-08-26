""" Training script for the agent """
from functools import partial
from pathlib import Path
from typing import Optional

import plac
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.algos.pg.a2c import A2C
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.utils.logging.context import logger_context

import utils
from machine_reading.ie import RedisWrapper
from machine_reading.ir.es import QASCIndexSearcher
from rl.aux import FocusedReadingTrajInfo
from rl.env import RlpytEnv, QASCInstanceFactory
from rl.models import FFFRMedium, FFFRLarge, FFFRExtraLarge


# run_id
# name
# log_dir
# snapshot_mode
# seed
# use_embeddings
# num_top_entities
# dataset_path
# batch_size
# num_steps
# log_interval_steps
# config_path

@plac.pos("slot_affinity_code", type=str)
@plac.pos("run_id", type=int)
@plac.pos("name")
@plac.pos("log_dir")
@plac.pos("snapshot_mode", choices=["all", "last"])
@plac.pos("seed", type=int)
@plac.pos("decorrelation_steps", type=int)
@plac.pos("t_steps", type=int)
@plac.pos("network_size", type=str, choices=['medium', 'large', 'xl'])
@plac.pos('embeddings_dropout', type=float)
@plac.pos("num_top_entities", type=int)
@plac.pos("dataset_path", type=Path)
@plac.pos("batch_size", type=int)
@plac.pos("num_steps", type=int)
@plac.pos("log_interval_steps", type=int)
@plac.flg("use_embeddings")
@plac.flg("reward_shaping")
@plac.flg("generational_ranking")
@plac.opt("config_path", type=Path)
def build_and_train(slot_affinity_code: str,
                    log_dir: str,
                    run_id: int,
                    name: str,
                    snapshot_mode: str,
                    seed: int,
                    decorrelation_steps: int,
                    t_steps: int,
                    network_size: str,
                    reward_shaping: bool,
                    generational_ranking: bool,
                    use_embeddings: bool,
                    embeddings_dropout: float,
                    num_top_entities: int,
                    dataset_path: Path,
                    batch_size: int,
                    num_steps: int,
                    log_interval_steps: int,
                    config_path: Optional[Path]):
    # Read the config file
    config = utils.read_config(config_path)
    train_config = config['rl_train']
    files = config['files']

    train_path = Path(files['train_file'])
    dev_path = Path(files['dev_file'])
    lucene_index_dir = files['lucene_index_dir']

    es = QASCIndexSearcher()
    redis = RedisWrapper()

    if slot_affinity_code.lower() == "none":
        slot_affinity_code = utils.get_affinity(config)

    affinity = affinity_from_code(slot_affinity_code)

    rng = utils.build_rng(seed)

    training_factory = QASCInstanceFactory.from_json(train_path, use_embeddings, num_top_entities,
                                                     es, redis, rng.randint(0, 1000))

    testing_factory = QASCInstanceFactory.from_json(dev_path, use_embeddings, num_top_entities,
                                                    es, redis, rng.randint(0, 1000))

    # Share the context data to avoid unnecessary redundancies
    testing_factory.nlp = training_factory.nlp
    # testing_factory.vector_space = training_factory.vector_space
    # testing_factory.topics = training_factory.topics
    # testing_factory.tfidf = training_factory.tfidf
    # testing_factory.index = testing_factory.index
    # testing_factory.inverted_index = testing_factory.inverted_index

    training_env_params = {
        'environment_factory': training_factory,
        'do_reward_shaping': reward_shaping
    }

    testing_env_params = {
        'environment_factory': testing_factory,
        'do_reward_shaping': False  # We shouldn't shape the reward on dev/testing
    }

    throwaway_env = training_factory()

    model_params = {
        "input_shape": throwaway_env.observe.shape[0],
        "action_space_size": 10,
        "use_embeddings": use_embeddings,
        "entity_dropout": embeddings_dropout
    }

    traj_cls = partial(FocusedReadingTrajInfo)

    sampler = SerialSampler(
        EnvCls=RlpytEnv,
        TrajInfoCls=traj_cls,
        env_kwargs=training_env_params,
        eval_env_kwargs=testing_env_params,
        batch_T=t_steps,
        batch_B=batch_size,
        max_decorrelation_steps=decorrelation_steps,
        eval_n_envs=8,#5//len(testing_factory.problems),
        eval_max_steps=100,#len(testing_factory.problems)*10,
        eval_max_trajectories=10,
    )

    # Resolve the correct network size and params
    network_size = network_size.lower()
    if network_size == 'medium':
        network_cls = FFFRMedium
    elif network_size == 'large':
        network_cls = FFFRLarge
    elif network_size == 'xl':
        network_cls = FFFRExtraLarge
    else:
        raise ValueError(f"Unrecognized network size: {network_size}")

    algo = A2C()  # Run with defaults.
    agent = CategoricalPgAgent(ModelCls=network_cls, model_kwargs=model_params)
    runner = MinibatchRl(
        # warm_up_itr=1000,
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=num_steps,
        log_interval_steps=log_interval_steps,
        affinity=affinity,
    )
    config = dict()

    with logger_context(log_dir, run_id, name, config, snapshot_mode=snapshot_mode, override_prefix=True):
        runner.train()


if __name__ == "__main__":
    plac.call(build_and_train)

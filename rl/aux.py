import time

import numpy as np
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.samplers.collections import TrajInfo
from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter

from actions import QueryType


class FocusedReadingTrajInfo(TrajInfo):
    """TrajInfo class for use with Atari Env, to store raw game score separate
    from clipped reward signal."""

    def __init__(self, num_entities, **kwargs):
        super().__init__(**kwargs)
        self.Papers = 0
        self.Success = False  # True is finding the path
        for ix in range(num_entities):
            self.__dict__[f'QueryAnd_{ix}'] = 0
            self.__dict__[f'QueryOr_{ix}'] = 0
            self.__dict__[f'QuerySingleton_{ix}'] = 0

        self._num_entities = num_entities

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        self.Papers += getattr(env_info, "papers", 0)
        self.Success = env_info.outcome
        qt = env_info.query_type
        e_ix = env_info.query_entity

        if qt == QueryType.And.value:
            self.__dict__[f'QueryAnd_{e_ix}'] += 1
        elif qt == QueryType.Or.value:
            self.__dict__[f'QueryOr_{e_ix}'] += 1
        elif qt == QueryType.Singleton.value:
            self.__dict__[f'QuerySingleton_{e_ix}'] += 1

        if done:
            num_queries = 0
            for ix in range(self._num_entities):
                num_queries += self.__dict__[f'QueryAnd_{ix}']
                num_queries += self.__dict__[f'QueryOr_{ix}']
                num_queries += self.__dict__[f'QuerySingleton_{ix}']

            for ix in range(self._num_entities):
                self.__dict__[f'QueryAnd_{ix}'] /= num_queries
                self.__dict__[f'QueryOr_{ix}'] /= num_queries
                self.__dict__[f'QuerySingleton_{ix}'] /= num_queries


class MinibatchRlEarlyStop(MinibatchRlEval):

    def __init__(self, warm_up_itr: int, *args, **kwargs):
        super(MinibatchRlEarlyStop, self).__init__(*args, **kwargs)
        self._warm_up_itr = warm_up_itr
        self._eval_returns = list()
        self._best_avg_return = None

    def train(self):
        """
        Performs startup, evaluates the initial agent, then loops by
        alternating between ``sampler.obtain_samples()`` and
        ``algo.optimize_agent()``.  Pauses to evaluate the agent at the
        specified log interval.
        """
        n_itr = self.startup()
        with logger.prefix(f"itr #0 "):
            eval_traj_infos, eval_time = self.evaluate_agent(0)
            self.log_diagnostics(0, eval_traj_infos, eval_traj_infos, eval_time)
        for itr in range(n_itr):
            logger.set_iteration(itr)
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(itr, samples)
                self.store_diagnostics(itr, traj_infos, opt_info)
                if (itr + 1) % self.log_interval_itrs == 0:
                    eval_traj_infos, eval_time = self.evaluate_agent(itr)
                    self.log_diagnostics(itr, traj_infos, eval_traj_infos, eval_time)
                    avg_return = np.average([e['Return'] for e in eval_traj_infos])
                    if self._best_avg_return is None:
                        self._best_avg_return = avg_return
                    self._eval_returns.append(avg_return)
                    if itr > self._warm_up_itr and self.ran_out_of_patience(n=30):
                        logger.log(f"Early stopping on  itr #{itr}")
                        break
        self.shutdown()

    def ran_out_of_patience(self, n=10) -> bool:
        """ Returns True to trigger early stopping when no improvement has been done on the evaluation set for the
        last n evaluations """

        avg_returns = self._eval_returns
        latest_avg_returns = avg_returns[-n:]

        if len(latest_avg_returns) == n:
            oldest = latest_avg_returns[0]
            if oldest > max(latest_avg_returns[1:]):
                return True
            else:
                return False
        else:
            return False

    def log_diagnostics(self, itr, train_traj_infos=None, eval_traj_infos=None, eval_time=0, is_training=False, prefix='Diagnostics/'):
        """
        Write diagnostics (including stored ones) to csv via the logger.
        """
        if itr > 0:
            self.pbar.stop()
        if itr >= self.min_itr_learn - 1:
            self.save_itr_snapshot(itr)
        new_time = time.time()
        self._cum_time = new_time - self._start_time
        train_time_elapsed = new_time - self._last_time - eval_time
        new_updates = self.algo.update_counter - self._last_update_counter
        new_samples = (self.sampler.batch_size * self.world_size *
                       self.log_interval_itrs)
        updates_per_second = (float('nan') if itr == 0 else
                              new_updates / train_time_elapsed)
        samples_per_second = (float('nan') if itr == 0 else
                              new_samples / train_time_elapsed)
        replay_ratio = (new_updates * self.algo.batch_size * self.world_size /
                        new_samples)
        cum_replay_ratio = (self.algo.batch_size * self.algo.update_counter /
                            ((itr + 1) * self.sampler.batch_size))  # world_size cancels.
        cum_steps = (itr + 1) * self.sampler.batch_size * self.world_size

        with logger.tabular_prefix(prefix):
            if is_training:
                logger.record_tabular('Type', 'Train')
            else:
                logger.record_tabular('Type', 'Test')
            if self._eval:
                logger.record_tabular('CumTrainTime',
                                      self._cum_time - self._cum_eval_time)  # Already added new eval_time.
            logger.record_tabular('Iteration', itr)
            logger.record_tabular('CumTime (s)', self._cum_time)
            logger.record_tabular('CumSteps', cum_steps)
            logger.record_tabular('CumCompletedTrajs', self._cum_completed_trajs)
            logger.record_tabular('CumUpdates', self.algo.update_counter)
            logger.record_tabular('StepsPerSecond', samples_per_second)
            logger.record_tabular('UpdatesPerSecond', updates_per_second)
            logger.record_tabular('ReplayRatio', replay_ratio)
            logger.record_tabular('CumReplayRatio', cum_replay_ratio)
        with logger.tabular_prefix("Train/"):
            self._log_infos(train_traj_infos)
        self._log_infos(eval_traj_infos)
        logger.dump_tabular(with_prefix=False)

        self._last_time = new_time
        self._last_update_counter = self.algo.update_counter
        if itr < self.n_itr - 1:
            logger.log(f"Optimizing over {self.log_interval_itrs} iterations.")
            self.pbar = ProgBarCounter(self.log_interval_itrs)

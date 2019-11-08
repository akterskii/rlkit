import abc
import pickle

import gtimer as gt
from rlkit.core import logger, eval_util
from rlkit.core.rl_algorithm import BaseRLAlgorithm, _get_epoch_timings
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.data_management.env_replay_buffer import DeadEndEnvReplayBuffer
from rlkit.samplers.data_collector import PathCollector, MdpEvaluationWithDanger


class BatchRLDangerAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            output_fname,
            trainer, #:Trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: MdpEvaluationWithDanger,
            replay_buffer: ReplayBuffer,
            replay_buffer_danger: DeadEndEnvReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_eps_for_evaluation,
            evaluation_after_steps,
            batch_danger_size=None,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )

        self.replay_buffer_danger = replay_buffer_danger
        self.batch_size = batch_size
        if batch_danger_size is not None:
            self.batch_danger_size = batch_danger_size
        else:
            self.batch_danger_size = max(batch_size // 2, 1)
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        # self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.num_eps = num_eps_for_evaluation
        self.evaluation_after_steps = evaluation_after_steps
        self.output_fname = output_fname

    def _log_stats(self, epoch, solved=False):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        if not solved:
            """
            Replay Buffer
            """
            logger.record_dict(
                self.replay_buffer.get_diagnostics(),
                prefix='replay_buffer/'
            )
            logger.record_dict(
                self.replay_buffer_danger.get_diagnostics(),
                prefix='replay_buffer_danger/'
            )
        if not solved:
            """
            Trainer
            """
            logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')
        if not solved:
            """
            Exploration
            """
            logger.record_dict(
                self.expl_data_collector.get_diagnostics(),
                prefix='exploration/'
            )
            expl_paths = self.expl_data_collector.get_epoch_paths()
            if hasattr(self.expl_env, 'get_diagnostics'):
                logger.record_dict(
                    self.expl_env.get_diagnostics(expl_paths),
                    prefix='exploration/',
                )
            logger.record_dict(
                eval_util.get_generic_path_information(expl_paths),
                prefix="exploration/",
            )

        """
        Evaluation
        """
        logger.record_dict(
            self.eval_data_collector.get_diagnostics(),
            prefix='evaluation/',
        )
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, 'get_diagnostics'):
            logger.record_dict(
                self.eval_env.get_diagnostics(eval_paths),
                prefix='evaluation/',
            )
        logger.record_dict(
            eval_util.get_generic_path_information(eval_paths),
            prefix="evaluation/",
        )

        """
        Misc
        """
        gt.stamp('logging')
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    def _end_epoch(self, epoch, solved=False):
        snapshot = self._get_snapshot()
        logger.save_itr_params(epoch, snapshot)
        gt.stamp('saving')
        self._log_stats(epoch, solved=solved)

        self.eval_data_collector.end_epoch(epoch)
        if not solved:
            self.expl_data_collector.end_epoch(epoch)
            self.replay_buffer.end_epoch(epoch)
            self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def save(self):
        with open(self.output_fname, "wb") as output_file:
            pickle.dump(self, output_file)

    def _train(self):
        if self.min_num_steps_before_training > 0:

            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
                random_exploration=True
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.replay_buffer_danger.add_paths(init_expl_paths)
            # for p in init_expl_paths:
            #    print("plen: {} last reward: {}".format(len(p['actions']), p['terminals'][-1]))
            # print("Dead size: ",self.replay_dead_buffer._size)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            solved = False
            
            if self.replay_buffer.num_steps_can_sample() >= self.evaluation_after_steps:
                _, solved = self.eval_data_collector.collect_new_paths(
                    max_path_length=self.max_path_length,
                    num_eps=self.num_eps,
                )
            else:
                # temporary quick evaluation
                self.eval_data_collector.collect_new_paths(
                    max_path_length=1000,
                    num_eps=1,
                )
            gt.stamp('evaluation sampling')

            if solved:
                self._end_epoch(epoch, solved=True)
                return True

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )

                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                self.replay_buffer_danger.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):

                    batch_normal = self.replay_buffer.random_batch(self.batch_size)
                    batch_danger = self.replay_buffer_danger.random_batch(self.batch_danger_size)
                    # sample 'safe' data for class balances
                    batch_safe = self.replay_buffer.random_batch(self.batch_danger_size)

                    # normal  - standard data for trining
                    # dead - data from end of the pathes with assigned probability of death
                    # safe - data from ordinary replay buffer, that is 'safe' with high probability
                    batch_full = {'normal': batch_normal, 'danger': batch_danger, 'safe': batch_safe}
                    self.trainer.train(batch_full)

                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)

        return False

import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.data_management.env_replay_buffer import DeadEndEnvReplayBuffer
from rlkit.samplers.data_collector import PathCollector


class BatchRLDeadAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer, #:Trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            replay_dead_buffer: DeadEndEnvReplayBuffer,
            batch_size,
            batch_dead_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
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

        self.replay_dead_buffer = replay_dead_buffer
        self.batch_size = batch_size
        self.batch_dead_size = batch_dead_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

    def _train(self):
        if self.min_num_steps_before_training > 0:
            # TODO add a random sampling
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.replay_dead_buffer.add_paths(init_expl_paths)
            for p in init_expl_paths:
                print("plen: {} last reward: {}".format(len(p['actions']), p['terminals'][-1]))
            print("Dead size: ",self.replay_dead_buffer._size)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                self.replay_dead_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    # TODO improve data labeling for train_dead_safe_data
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    #print(self.batch_dead_size, self.replay_dead_buffer._size)
                    train_dead_dead_data = self.replay_dead_buffer.random_batch(
                        self.batch_dead_size)
                    # sample 'safe' data for class balances
                    train_dead_safe_data = self.replay_buffer.random_batch(
                        self.batch_dead_size)
                    # normal  - standard data for trining
                    # dead - data from end of the pathes with assigned probability of death
                    # safe - data from ordinary replay buffer, that is 'safe' with high probability
                    train = {'normal': train_data, 'dead': train_dead_dead_data, 'safe': train_dead_safe_data}
                    self.trainer.train(train)

                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._end_epoch(epoch)

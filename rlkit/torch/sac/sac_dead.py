from collections import OrderedDict

import numpy as np
# import torch
# import torch.optim as optim
from torch import nn as nn

from typing import Iterable
from torch.optim.optimizer import Optimizer

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.radam import RAdam
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.core import np_to_pytorch_batch


class DeadTrainer(TorchTrainer):
    def __init__(self,
                 qf_dead,
                 policy_dead,

                 pass_per_iteration=1,
                 policy_lr=1e-3,
                 qf_lr=1e-3,
                 optimizer_class=RAdam,  # optim.Adam

                 policy_update_period=1,
                 plotter=None,
                 render_eval_paths=False,
                 ):
        super().__init__()
        self.policy_and_target_update_period = policy_update_period
        self.qf_dead = qf_dead
        self.policy_dead = policy_dead

        self.qf_dead_optimizer = optimizer_class(
            self.qf_dead.parameters(),
            lr=qf_lr,
        )

        self.policy_dead_optimizer = optimizer_class(
            self.policy_dead.parameters(),
            lr=policy_lr,
        )

        self.pass_per_iteration = pass_per_iteration
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.qf_dead_criterion = nn.MSELoss()

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train(self, np_batch, np_batch_dead):
        self._num_train_steps += 1
        # for k in np_batch.keys():
        #     print(k)
        np_full_batch = {k:np.concatenate((np_batch[k], np_batch_dead[k])) for k in np_batch.keys()}
        full_batch = np_to_pytorch_batch(np_full_batch)
        batch_dead = np_to_pytorch_batch(np_batch_dead)
        self.train_from_torch(full_batch, batch_dead)

    def train_from_torch(self, batch, batch_dead):
        """
        :param batch: dictionary with two elements:
        :return:
        """
        #  for policy training
        obs_dead = batch_dead['observations']

        # for q-function training
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']

        cur_predictions = self.qf_dead(obs, actions)

        qf_dead_loss = self.qf_dead_criterion(cur_predictions, terminals)

        """
        Update Q-Network
        """
        self.qf_dead_optimizer.zero_grad()
        qf_dead_loss.backward()
        self.qf_dead_optimizer.step()

        """
        Update policy-network
        """
        # TODO check if it worth to train policy on danger states only

        policy_dead_loss = policy_actions = None
        if self._n_train_steps_total % self.policy_and_target_update_period == 0:
            policy_actions = self.policy_dead(obs_dead)
            policy_dead_loss = self.qf_dead(obs_dead, policy_actions).mean()

            self.policy_dead_optimizer.zero_grad()
            policy_dead_loss.backward()
            self.policy_dead_optimizer.step()

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            if policy_dead_loss is None:
                policy_actions = self.policy_dead(obs)
                policy_dead_loss = self.qf_dead(obs, policy_actions).mean()
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """

            self.eval_statistics['QF dead Loss'] = np.mean(ptu.get_numpy(qf_dead_loss))
            self.eval_statistics['Policy dead Loss'] = np.mean(ptu.get_numpy(
                policy_dead_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF dead Predictions',
                ptu.get_numpy(cur_predictions),
            ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'QF dead Predictions danger',
            #     ptu.get_numpy(cur_predicts_danger),
            # ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy  Actions',
                ptu.get_numpy(policy_actions),
            ))

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self) -> Iterable[nn.Module]:
        return [
            self.qf_dead,
            self.policy_dead,
        ]

    @property
    def optimizers(self) -> Iterable[Optimizer]:
        return [
            self.qf_dead_optimizer,
            self.policy_dead_optimizer
        ]

    def get_snapshot(self):
        return dict(
            qf_dead=self.qf_dead,
            policy_dead=self.policy_dead,
            qf_dead_optimizer=self.qf_dead_optimizer,
            policy_dead_optimizer=self.policy_dead_optimizer,
        )


class SACDeadTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            qf_dead,
            global_policy,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=RAdam,  # optim.Adam
            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        super().__init__()
        sac_trainer_params = {
            'discount': discount,
            'reward_scale': reward_scale,
            'policy_lr': policy_lr,
            'optimizer_class': optimizer_class,
            'target_update_period': target_update_period,
            'plotter': plotter,
            'render_eval_paths': render_eval_paths,
            'use_automatic_entropy_tuning': use_automatic_entropy_tuning,
            'target_entropy': target_entropy
            }
        dead_trainer_params = {
            'pass_per_iteration': 1,
        }
        self.sacTrainer = SACTrainer(
            env=env,
            policy=global_policy.tanhGaussianPolicy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **sac_trainer_params
        )

        self.deadTrainer = DeadTrainer(
            policy_dead=global_policy.deadPredictionPolicy,
            qf_dead=qf_dead,
            **dead_trainer_params
        )

        ###################################

        self.env = env
        self.policy = global_policy

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train(self, batch):
        """
        :param batch: dict with 3 fields 'normal', 'safe' and 'danger'
         'normal' part is using for
        :return:
        """
        batch_normal = batch['normal']
        #batch_dead = {key: np.concatenate((batch['dead'][key], batch['safe'][key])) for key in batch['dead']}

        self.sacTrainer.train(batch_normal)

        self.deadTrainer.train(np_batch=batch['safe'], np_batch_dead=batch['dead'])

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics = OrderedDict({**self.sacTrainer.eval_statistics, **self.deadTrainer.eval_statistics})

        self._n_train_steps_total += 1

    def train_from_torch(self, batch):
        assert 'Blank method. Should not be called' == ''

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self.sacTrainer.end_epoch(epoch)
        self.deadTrainer.end_epoch(epoch)
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return self.sacTrainer.networks + self.deadTrainer.networks

    @property
    def optimizers(self) -> Iterable[Optimizer]:
        return self.sacTrainer.optimizers + self.deadTrainer.optimizers

    def get_snapshot(self):
        d1 = self.sacTrainer.get_snapshot()
        d2 = self.deadTrainer.get_snapshot()
        return {**d1, **d2}


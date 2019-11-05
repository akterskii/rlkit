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
from rlkit.torch.core import np_to_pytorch_batch


class DangerTrainer(TorchTrainer):
    def __init__(self,
                 qf_danger,
                 policy_danger,

                 policy_lr=1e-3,
                 qf_lr=1e-3,
                 optimizer_class=RAdam,  # optim.Adam

                 policy_update_period=1,
                 plotter=None,
                 render_eval_paths=False,
                 ):
        super().__init__()
        self.policy_and_target_update_period = policy_update_period
        self.qf_danger = qf_danger
        self.policy_danger = policy_danger

        self.qf_danger_optimizer = optimizer_class(
            self.qf_danger.parameters(),
            lr=qf_lr,
        )

        self.policy_danger_optimizer = optimizer_class(
            self.policy_danger.parameters(),
            lr=policy_lr,
        )


        self.plotter = plotter
        self.render_eval_paths = render_eval_paths
        self.qf_danger_criterion = nn.MSELoss()

        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train(self, np_batch_dict):
        """
        :param np_batch_dict: dict with 'safe' and 'danger' keys with batches
        :return:
        """
        self._num_train_steps += 1

        # concat data for danger and safe transitions
        np_full_batch = {k: np.concatenate((np_batch_dict['safe'][k], np_batch_dict['danger'][k])) for k in
                         np_batch_dict['safe'].keys()}

        # convert to pytorch
        torch_batch_full = np_to_pytorch_batch(np_full_batch)
        torch_batch_danger = np_to_pytorch_batch(np_batch_dict['safe'])

        # pack it to dict
        torch_batch_dict = {'full': torch_batch_full, 'danger': torch_batch_danger}
        self.train_from_torch(torch_batch_dict)

    def train_from_torch(self, torch_batch_dict):
        """
        :param torch_batch_dict: dictionary with two elements: 'danger' and 'full'
        :return:
        """
        torch_batch_danger = torch_batch_dict['danger']
        torch_batch_full = torch_batch_dict['full']

        #  for policy training
        observations_danger = torch_batch_danger['observations']

        # for q-function training
        terminals = torch_batch_full['terminals']
        obs = torch_batch_full['observations']
        actions = torch_batch_full['actions']

        cur_predictions = self.qf_danger(obs, actions)

        qf_danger_loss = self.qf_danger_criterion(cur_predictions, terminals)

        """
        Update Q-Network
        """
        self.qf_danger_optimizer.zero_grad()
        qf_danger_loss.backward()
        self.qf_danger_optimizer.step()

        """
        Update policy-network
        """
        policy_danger_loss = policy_actions = None
        if self._n_train_steps_total % self.policy_and_target_update_period == 0:
            policy_actions = self.policy_danger(observations_danger)
            policy_danger_loss = self.qf_danger(observations_danger, policy_actions).mean()

            self.policy_danger_optimizer.zero_grad()
            policy_danger_loss.backward()
            self.policy_danger_optimizer.step()

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            if policy_danger_loss is None:
                policy_actions = self.policy_danger(obs)
                policy_danger_loss = self.qf_danger(obs, policy_actions).mean()
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """

            self.eval_statistics['QF danger Loss'] = np.mean(ptu.get_numpy(qf_danger_loss))
            self.eval_statistics['Policy danger Loss'] = np.mean(ptu.get_numpy(
                policy_danger_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF danger Predictions',
                ptu.get_numpy(cur_predictions),
            ))
            # self.eval_statistics.update(create_stats_ordered_dict(
            #     'QF danger Predictions danger',
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
            self.qf_danger,
            self.policy_danger,
        ]

    @property
    def optimizers(self) -> Iterable[Optimizer]:
        return [
            self.qf_danger_optimizer,
            self.policy_danger_optimizer
        ]

    def get_snapshot(self):
        return dict(
            qf_danger=self.qf_danger,
            policy_danger=self.policy_danger,
            qf_danger_optimizer=self.qf_danger_optimizer,
            policy_danger_optimizer=self.policy_danger_optimizer,
        )


class DangerTrainerFull(TorchTrainer):
    def __init__(
            self,
            env,

            normal_trainer: TorchTrainer,
            danger_trainer: DangerTrainer,

            plotter=None,
            render_eval_paths=False,
    ):
        super().__init__()

        self.normal_trainer = normal_trainer
        self.danger_trainer = danger_trainer

        ###################################

        self.env = env

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
        np_batch_normal = batch['normal']
        np_batch_dict = {'safe': batch['safe'], 'danger': batch['danger']}
        self.normal_trainer.train(np_batch_normal)
        self.danger_trainer.train(np_batch_dict)

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics = OrderedDict({**self.normal_trainer.eval_statistics,
                                                **self.danger_trainer.eval_statistics})

        self._n_train_steps_total += 1

    def train_from_torch(self, batch):
        assert 'Blank method. Should not be called' == ''

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self.normal_trainer.end_epoch(epoch)
        self.danger_trainer.end_epoch(epoch)
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return self.normal_trainer.networks + self.danger_trainer.networks

    @property
    def optimizers(self) -> Iterable[Optimizer]:
        return self.normal_trainer.optimizers + self.danger_trainer.optimizers

    def get_snapshot(self):
        d1 = self.normal_trainer.get_snapshot()
        d2 = self.danger_trainer.get_snapshot()
        return {**d1, **d2}


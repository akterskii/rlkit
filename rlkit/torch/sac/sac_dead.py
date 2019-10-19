from collections import OrderedDict

import numpy as np
# import torch
# import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.radam import RAdam
from rlkit.torch.sac.sac import SACTrainer


class DeadTrainer(TorchTrainer):
    def __init__(self,
                 qf_dead,
                 policy_dead,

                 pass_per_iteration=1,
                 policy_lr=1e-3,
                 qf_lr=1e-3,
                 optimizer_class=RAdam,  # optim.Adam

                 policy_and_target_update_period=1,
                 plotter=None,
                 render_eval_paths=False,
                 ):
        super().__init__()
        self.policy_and_target_update_period = policy_and_target_update_period
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

    def train_from_torch(self, batch):
        good_batch = batch['good']
        danger_batch = batch['danger']

        terminals_good = good_batch['terminals']
        obs_good = good_batch['observations']
        actions_good = good_batch['actions']

        terminals_danger = danger_batch['terminals']
        obs_danger = danger_batch['observations']
        actions_danger = danger_batch['actions']

        cur_predicts_good = self.qf_dead(obs_good, actions_good)
        cur_predicts_danger = self.qf_dead(obs_danger, actions_danger)

        qf_dead_loss = 1 / 2 * (self.qf_dead_criterion(cur_predicts_danger, terminals_danger)
                                + self.qf_dead_criterion(cur_predicts_good, terminals_good))

        """
        Update Q-Network
        """
        self.qf_dead_optimizer.zero_grad()
        qf_dead_loss.backward()
        self.qf_dead_optimizer.step()

        """
        Update policy-network
        """
        policy_dead_loss = policy_actions = None
        if self._n_train_steps_total % self.policy_and_target_update_period == 0:
            policy_actions = self.policy_dead(obs_danger)
            policy_dead_loss = self.qf_dead(obs_danger, policy_actions).mean()
            self.policy_dead_optimizer.zero_grad()
            policy_dead_loss.backward()
            self.policy_dead_optimizer.step()

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            if policy_dead_loss is None:
                policy_actions = self.policy_dead(obs_danger)
                policy_dead_loss = self.qf_dead(obs_danger, policy_actions).mean()
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """

            self.eval_statistics['QF dead Loss'] = np.mean(ptu.get_numpy(qf_dead_loss))
            self.eval_statistics['Policy dead Loss'] = np.mean(ptu.get_numpy(
                policy_dead_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF dead Predictions good',
                ptu.get_numpy(cur_predicts_good),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF dead Predictions danger',
                ptu.get_numpy(cur_predicts_danger),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy dead Action',
                ptu.get_numpy(policy_actions),
            ))

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.qf_dead,
            self.policy_dead,
        ]

    def get_snapshot(self):
        return dict(
            qf_dead=self.qf_dead,
            policy_dead=self.policy_dead,
        )


class SACDeadTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            qf_dead,
            policy_dead,

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
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **sac_trainer_params
        )

        self.deadTrainer = DeadTrainer(
            policy_dead=policy_dead,
            qf_dead=qf_dead,
            **dead_trainer_params
        )

        ###################################

        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.qf_dead = qf_dead,
        self.policy_dead = policy_dead
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )

        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )

        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        batch_normal = batch['normal']
        batch_dead = {'dead': batch['dead'], 'safe': batch['safe']}

        self.sacTrainer.train_from_torch(batch_normal)
        self.deadTrainer.train_from_torch(batch_dead)

        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics = OrderedDict({**self.sacTrainer.eval_statistics, **self.deadTrainer.eval_statistics})

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self.sacTrainer.end_epoch(epoch)
        self.deadTrainer.end_epoch(epoch)
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return self.sacTrainer.networks + self.deadTrainer.networks

    def get_snapshot(self):
        d1 = self.sacTrainer.get_snapshot()
        d2 = self.deadTrainer.get_snapshot()
        return {**d1, **d2}


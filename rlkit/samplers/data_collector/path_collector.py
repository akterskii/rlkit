from collections import deque, OrderedDict

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.rollout_functions import rollout, multitask_rollout
from rlkit.samplers.data_collector.base import PathCollector
from rlkit.torch.sac.policies import DangerPolicyCounterWrapper
from rlkit.envs.wrappers import EnvWithActionRepeat
import numpy as np


class MdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
            random_exploration=False,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                random_exploration=random_exploration
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
        )


class MdpPathCollectorWithDanger(MdpPathCollector):
    def __init__(self,
                 env,
                 policy: DangerPolicyCounterWrapper,
                 max_num_epoch_paths_saved=None,
                 render=False,
                 render_kwargs=None,):
        super().__init__(env, policy, max_num_epoch_paths_saved, render, render_kwargs)

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats['danger_updates'] = self._policy.get_updates_count()
        return stats

    def collect_new_episodes(
            self,
            max_path_length,
            episodes_amount,
    ):
        paths = []
        num_steps_collected = 0
        for _ in range(episodes_amount):
            path = rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length,
            )
            path_len = len(path['actions'])
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)

        return paths, num_steps_collected


class MdpEvaluationWithDanger(MdpPathCollectorWithDanger):
    def __init__(self,
                 env,
                 policy_danger,
                 terminal_reward,
                 reward_to_pass,
                 pass_criterion_name=None,
                 max_num_epoch_paths_saved=None,
                 render=False,
                 render_kwargs=None):
        super().__init__(env=env,
                         policy=policy_danger,
                         max_num_epoch_paths_saved=max_num_epoch_paths_saved,
                         render=render,
                         render_kwargs=render_kwargs
                         )
        self.terminal_reward = terminal_reward
        self.reward_to_pass = reward_to_pass
        if pass_criterion_name is None:
            self.pass_criterion_name = 'min_reward_gt_threshold'
        else:
            self.pass_criterion_name = pass_criterion_name

    def pass_criterion(self, returns, lengths, terminals, last_rewards, max_path_length):
        # every return is higher than threshold
        if self.pass_criterion_name == 'min_reward_gt_threshold':
            return np.min(returns) >= self.reward_to_pass
        # every episode finished without dead
        elif self.pass_criterion_name == 'reach_the_end':
            return (np.max(lengths) < max_path_length) and \
                   (np.min(last_rewards) > self.terminal_reward) and \
                   (sum(terminals) == len(terminals))
        else:
            raise NotImplementedError('Passing criterion with this name not implemented')


    def collect_new_paths(
        self,
        max_path_length,
        num_eps
    ):
        paths = []
        ep_collected = 0
        fails = 0
        num_steps_collected = 0
        while ep_collected < num_eps:
            path = rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length
            )
            path_len = len(path['actions'])
            ep_collected += 1
            paths.append(path)
        self._num_steps_total += num_steps_collected
        self._num_paths_total += len(paths)
        self._epoch_paths.extend(paths)

        last_rewards = [path["rewards"][-1] for path in paths]
        returns = [sum(path["rewards"]) for path in paths]
        lengths = [len(path["actions"]) for path in paths]
        terminals = [path['terminals'][-1] for path in paths]

        #  passed criterion
        solved = False

        if self.pass_criterion(returns, lengths, terminals, last_rewards, max_path_length):
            print("Solved")
            solved = True

        # # reach the end
        # def criterion(path):
        #     return path['terminals'] and len(path['rewards']) <= max_path_length and \
        #            path['reward'][-1] != self.terminal_reward
        # finished = sum([1 if criterion(path) else 0 for path in paths])
        # if finished == len(paths):
        #     solved = True
        return paths, solved


class GoalConditionedPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            observation_key='observation',
            desired_goal_key='desired_goal',
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._render = render
        self._render_kwargs = render_kwargs
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = multitask_rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
                observation_key=self._observation_key,
                desired_goal_key=self._desired_goal_key,
                return_dict_obs=True,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            num_steps_collected += path_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )

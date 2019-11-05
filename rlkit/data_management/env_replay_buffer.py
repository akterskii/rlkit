from gym.spaces import Discrete

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )


class DeadEndEnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None,
            probability_function='Linear',
            steps_to_end=10,
            threshold=0.7
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        :param env_info_sizes:
        :param probability_function:
        :param steps_to_end:
        :param threshold:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        self._probability_function = probability_function
        self._steps_to_end = steps_to_end
        self._threshold = threshold

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )
        self._terminals = np.zeros((max_replay_buffer_size, 1))

    def _get_probability_by_number(self, n):
        # 1-based numbering
        # n == self._steps_to_end corresponds to the last state-action pair, with prob == 1

        assert 0 < n <= self._steps_to_end

        # no probability function needed
        # also it handles case _steps_to_end == 1
        if self._steps_to_end == n:
            return 1

        if self._probability_function == 'Linear':
            return self._threshold + (1 - self._threshold) * (n - 1) / (self._steps_to_end - 1)
        else:
            raise NotImplementedError('Probability type not implemented in {}.'.format(self.__class__.__name__))

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )

    def add_paths(self, paths, terminal_reward=-100):
        # print('checgjh', len(paths))
        for path in paths:
            # print('len of cur path', len(path['rewards']), ' ', path['terminals'][-1], ' ', path['rewards'][-1])
            if path['terminals'][-1] and abs(path['rewards'][-1] - terminal_reward) < 10e-6:
                # print('add')
                self.add_path(path)

    def add_path(self, path):
        # rewrite it
        start_index = max(0, len(path['terminals']) - self._steps_to_end)
        n = self._steps_to_end

        for i in range(len(path['terminals']) - 1, start_index - 1, -1):
            # print('pos', i)
            cur_prob = self._get_probability_by_number(n)
            # print("cur prob: ", cur_prob)
            self.add_sample(path['observations'][i], path['actions'][i], path['rewards'][i], cur_prob,
                            path['next_observations'][i], env_info=None)
            n -= 1

    def get_all_items(self):
        print(len(self._observations))
        for i in range(self._size):
            print(self._observations[i], self._actions[i], self._rewards[i], self._terminals[i])

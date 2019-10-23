# from gym.envs.mujoco import HalfCheetahEnv
from gym.envs.box2d import BipedalWalkerHardcore

import gtimer as gt
from torch.nn import functional as F
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer, DeadEndEnvReplayBuffer

from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, DeadTanhPolicy
from rlkit.torch.sac.sac_dead import SACDeadTrainer
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy

from rlkit.torch.torch_rl_algorithm import TorchBatchRLDeadAlgorithm


def experiment(variant):
    env_class = variant['env_class']
    expl_env = NormalizedBoxEnv(env_class())
    eval_env = NormalizedBoxEnv(env_class())
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    # TODO pass terminal reward
    #  variant['env_terminal_reward']

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    sac_policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    qf_dead = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
        output_activation=F.sigmoid,
    )
    policy_dead = TanhMlpPolicy(
        input_size=obs_dim,
        hidden_sizes=[M, M],
        output_size=action_dim,
        #  **variant['policy_kwargs']
    )

    global_policy = DeadTanhPolicy(
        tanh_gaussian_policy=sac_policy,
        dead_prediction_policy=policy_dead,
        dead_prediction_qf=qf_dead,
        threshold=variant['threshold']
    )

    eval_policy = MakeDeterministic(global_policy)

    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )

    expl_path_collector = MdpPathCollector(
        expl_env,
        global_policy,
    )

    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )

    replay_dead_buffer = DeadEndEnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )

    trainer = SACDeadTrainer(
        env=eval_env,
        global_policy=global_policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        qf_dead=qf_dead,
        **variant['trainer_kwargs']
    )

    algorithm = TorchBatchRLDeadAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        replay_dead_buffer=replay_dead_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC dead",
        version="normal",
        env_class=BipedalWalkerHardcore,
        env_terminal_reward=-100,
        layer_size=256,
        replay_buffer_size=int(1E6),
        threshold=0.7,

        algorithm_kwargs=dict(
            num_epochs=3,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=500,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=100,
            batch_size=100,
            batch_dead_size= 50 # 2 times smaller than batch_size
        ),

        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger('SAC+dead', variant=variant, log_dir='d:/tmp2', to_file_only=False)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    gt.reset_root()  # for interactive restarts
    experiment(variant)

import gtimer as gt
import torch
from torch import optim
import copy
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import setup_logger

# Envs
from rlkit.envs.wrappers import NormalizedBoxEnv, EnvWithActionRepeat
from gym.envs.box2d import BipedalWalkerHardcore, BipedalWalker, LunarLanderContinuous
# from gym.envs.mujoco import HalfCheetahEnv

# Samplers
from rlkit.samplers.data_collector import MdpPathCollectorWithDanger
from rlkit.torch.ddpg.ddpg import DDPGTrainer
from rlkit.torch.samplers.path_collector import MdpEvaluationWithDangerTorch

# Buffers
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer, DeadEndEnvReplayBuffer

# Policies and Qfs
from rlkit.torch.sac.policies import TanhGaussianPolicy, DangerAndPolicy, DangerPolicyCounterWrapper, MakeDeterministic
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy

# Trainers
from rlkit.torch.danger import DangerTrainer, DangerTrainerFull
from rlkit.torch.sac.sac import SACTrainer

# Algorithms
from rlkit.torch.td3.td3 import TD3Trainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLDangerAlgorithm


def get_sac(evaluation_environment, parameters):
    """
    :param env - environment to get action shape
    :param parameters: dict with keys -
    hidden_sizes,
    sac_trainer_parameters
    :return: sac_policy, eval_policy, trainer
    """
    obs_dim = evaluation_environment.observation_space.low.size
    action_dim = evaluation_environment.action_space.low.size

    hidden_sizes_qf = parameters['hidden_sizes_qf']
    hidden_sizes_policy = parameters['hidden_sizes_policy']

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_qf,
    )

    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_qf,
    )

    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_qf,
    )

    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_qf,
    )

    sac_policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes_policy,
    )

    eval_policy = MakeDeterministic(sac_policy)

    trainer = SACTrainer(
        env=evaluation_environment,
        policy=sac_policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **parameters['trainer_params']
    )

    return sac_policy, eval_policy, trainer


def get_td3pg(evaluation_environment, parameters):
    """
    :param evaluation_environment:
    :param parameters:
    :return:
    """
    obs_dim = evaluation_environment.observation_space.low.size
    action_dim = evaluation_environment.action_space.low.size

    hidden_sizes_qf = parameters['hidden_sizes_qf']
    hidden_sizes_policy = parameters['hidden_sizes_policy']

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_qf,
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_qf,
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_qf,
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_qf,
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes_policy,
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes_policy,
    )
    es = GaussianStrategy(
        action_space=evaluation_environment.action_space,
        max_sigma=0.1,
        min_sigma=0.1,  # Constant sigma
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    trainer = TD3Trainer(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **parameters['trainer_params']
    )
    return exploration_policy, policy, trainer


def get_ddpg(evaluation_environment, parameters):
    obs_dim = evaluation_environment.observation_space.low.size
    action_dim = evaluation_environment.action_space.low.size
    hidden_sizes_qf = parameters['hidden_sizes_qf']
    hidden_sizes_policy = parameters['hidden_sizes_policy']

    qf = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_qf,
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes_policy,
    )
    target_qf = copy.deepcopy(qf)
    target_policy = copy.deepcopy(policy)

    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=OUStrategy(action_space=evaluation_environment.action_space),
        policy=policy,
    )

    trainer = DDPGTrainer(
        qf=qf,
        target_qf=target_qf,
        policy=policy,
        target_policy=target_policy,
        **parameters['trainer_params']
    )
    return exploration_policy, policy, trainer


def get_danger(env, parameters):
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    hidden_sizes_qf = parameters['hidden_sizes_qf']
    hidden_sizes_policy = parameters['hidden_sizes_policy']

    qf_danger = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_qf,
        output_activation=torch.sigmoid,
    )

    policy_danger = TanhMlpPolicy(
        input_size=obs_dim,
        hidden_sizes=hidden_sizes_policy,
        output_size=action_dim,
    )

    danger_trainer = DangerTrainer(qf_danger=qf_danger,
                                   policy_danger=policy_danger,
                                   **parameters['trainer_danger_params'])

    return policy_danger, qf_danger, danger_trainer


def experiment(variant):
    general_params = variant['general_params']
    environment_params = variant['environment_params']
    base_params = variant['base_params']
    danger_params = variant['danger_params']
    algorithm_params = variant['algorithm_params']

    env_class = environment_params['class_name']
    expl_env = EnvWithActionRepeat(NormalizedBoxEnv(env_class()), repeat_action=3)
    eval_env = NormalizedBoxEnv(env_class())

    algorithm_name = general_params['algorithm_name']
    if algorithm_name == 'DDPG':
        base_policy, base_policy_evaluation, base_trainer = get_ddpg(eval_env, base_params['ddpg'])
    elif algorithm_name == 'TD3PG':
        base_policy, base_policy_evaluation, base_trainer = get_td3pg(eval_env, base_params['td3pg'])
    elif algorithm_name == 'SAC':
        base_policy, base_policy_evaluation, base_trainer = get_sac(eval_env, base_params['sac'])
    else:
        raise NotImplementedError('Unknown algorithm')

    #######################  Danger  ############################
    policy_danger, qf_danger, danger_trainer = get_danger(eval_env, danger_params)

    #######################  Global  ############################
    global_policy_evaluation = DangerAndPolicy(
        policy_base=base_policy_evaluation,
        policy_danger=policy_danger,
        qf_danger_probability=qf_danger,
        threshold=danger_params['threshold']
    )

    global_policy_exploration = DangerAndPolicy(
        policy_base=base_policy,
        policy_danger=policy_danger,
        qf_danger_probability=qf_danger,
        threshold=danger_params['threshold']
    )

    expl_policy = DangerPolicyCounterWrapper(global_policy_exploration)
    eval_policy = DangerPolicyCounterWrapper(global_policy_evaluation)

    eval_path_collector = MdpEvaluationWithDangerTorch(
        eval_env,
        eval_policy,
        terminal_reward=environment_params['env_terminal_reward'],
        reward_to_pass=environment_params['reward_to_pass']
    )

    expl_path_collector = MdpPathCollectorWithDanger(expl_env, expl_policy)

    replay_buffer = EnvReplayBuffer(
        general_params['replay_buffer_size'],
        expl_env,
    )

    replay_buffer_danger = DeadEndEnvReplayBuffer(
        max_replay_buffer_size=general_params['replay_buffer_size'],
        env=expl_env,
        probability_function=danger_params['probability_function'],
        steps_to_end=danger_params['steps_to_end'],
        threshold=danger_params['threshold'],
    )

    trainer = DangerTrainerFull(
        env=eval_env,
        normal_trainer=base_trainer,
        danger_trainer=danger_trainer,
        plotter=None,
        render_eval_paths=False,
    )

    #############################################################
    algorithm = TorchBatchRLDangerAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        replay_buffer_danger=replay_buffer_danger,
        **algorithm_params
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        general_params=dict(
            algorithm_name="DDPG", #SAC TD3PG
            replay_buffer_size=int(1E6)
        ),

        environment_params=dict(
            class_name=BipedalWalker,  # BipedalWalker, BipedalWalkerHardcore,
            reward_to_pass=0,                 # 300            300
            env_terminal_reward=-200,         # -100           -100
        ),

        base_params=dict(
            sac=dict(
                hidden_sizes_qf=[256, 256],
                hidden_sizes_policy=[256, 256],
                trainer_params=dict(
                    discount=0.99,
                    soft_target_tau=5e-3,
                    target_update_period=1,
                    policy_lr=3E-4,
                    qf_lr=3E-4,
                    reward_scale=1,
                    use_automatic_entropy_tuning=True,
                ),
            ),
            td3pg=dict(
                hidden_sizes_qf=[400, 300],
                hidden_sizes_policy=[400, 300],
                trainer_params=dict(
                    target_policy_noise=0.2,
                    target_policy_noise_clip=0.5,

                    discount=0.99,
                    reward_scale=1.0,

                    policy_learning_rate=1e-3,
                    qf_learning_rate=1e-3,
                    policy_and_target_update_period=2,
                    tau=0.005,
                    qf_criterion=None,
                    optimizer_class=optim.Adam,
                )
            ),
            ddpg=dict(
                hidden_sizes_qf = [400, 300],
                hidden_sizes_policy=[400, 300],
                trainer_params=dict(
                    discount=0.99,
                    reward_scale=1.0,

                    policy_learning_rate=1e-4,
                    qf_learning_rate=1e-3,
                    qf_weight_decay=0,
                    target_hard_update_period=1000,
                    tau=1e-2,
                    use_soft_update=False,
                    qf_criterion=None,
                    policy_pre_activation_weight=0.,
                    optimizer_class=optim.Adam,

                    min_q_value=-np.inf,
                    max_q_value=np.inf,
                )
            )

        ),

        danger_params=dict(
            threshold=0.7,
            probability_function='Linear',
            hidden_sizes_qf=[400, 300],
            hidden_sizes_policy=[400, 300],
            steps_to_end=10,
            trainer_danger_params=dict(
                policy_lr=1e-3,
                qf_lr=1e-3,
                #optimizer_class=RAdam,  # optim.Adam
                policy_update_period=1,
                plotter=None,
                render_eval_paths=False,
            ),
        ),

        algorithm_params=dict(
            output_fname='D:\tmp2\out.txt',
            batch_size= 32, #256,
            max_path_length=700,
            num_epochs=2,
            num_expl_steps_per_train_loop=2000,
            num_trains_per_train_loop=200,
            num_eps_for_evaluation=1,
            evaluation_after_steps=200,
            batch_danger_size=None,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=100,
        ),
    )

    setup_logger('SAC+dead', variant=variant, log_dir='d:/tmp2', to_file_only=False)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    gt.reset_root()         # for interactive restarts
    experiment(variant)

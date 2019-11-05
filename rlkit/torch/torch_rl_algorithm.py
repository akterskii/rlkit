import abc
from collections import OrderedDict

from typing import Iterable
from torch import nn as nn
from torch.optim.optimizer import Optimizer
import torch

from rlkit.core.batch_rl_algorithm import BatchRLAlgorithm
from rlkit.core.batch_rl_dead_algorithm import BatchRLDangerAlgorithm
from rlkit.core.online_rl_algorithm import OnlineRLAlgorithm
from rlkit.core.trainer import Trainer
from rlkit.torch.core import np_to_pytorch_batch


class TorchOnlineRLAlgorithm(OnlineRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchBatchRLAlgorithm(BatchRLAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchBatchRLDangerAlgorithm(BatchRLDangerAlgorithm):
    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchTrainer(Trainer, metaclass=abc.ABCMeta):
    def __init__(self):
        self._num_train_steps = 0
        self.eval_statistics = OrderedDict()

    def train(self, np_batch):
        self._num_train_steps += 1
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(batch)

    def get_diagnostics(self):
        return OrderedDict([
            ('num train calls', self._num_train_steps),
        ])

    @abc.abstractmethod
    def train_from_torch(self, batch):
        pass

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[nn.Module]:
        pass

    @property
    @abc.abstractmethod
    def optimizers(self) -> Iterable[Optimizer]:
        pass

    def load_nets(self, path):
        checkpoint = torch.load(path)

        nets = self.networks
        optimizers = self.optimizers
        object_num = 0

        for net in nets:
            net.load_state_dict(checkpoint[object_num])
            object_num += 1
        for optimizer in optimizers:
            optimizer.load_state_dict(checkpoint[object_num])

    def save_nets(self, path):
        nets = self.networks
        optimizers = self.optimizers

        model_params = {}
        object_num = 0
        for net in nets:
            model_params[object_num] = net.state_dict()
            object_num += 1

        for optimizer in optimizers:
            model_params[object_num] = optimizer.state_dict()
            object_num += 1

        torch.save(model_params, path)
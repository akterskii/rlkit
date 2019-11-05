import torch
from rlkit.samplers.data_collector.path_collector import MdpEvaluationWithDanger


class MdpEvaluationWithDangerTorch(MdpEvaluationWithDanger):
    def collect_new_paths(
        self,
        max_path_length,
        num_eps
    ):
        with torch.no_grad():
            return super().collect_new_paths(max_path_length, num_eps)

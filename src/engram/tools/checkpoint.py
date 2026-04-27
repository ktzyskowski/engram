import os

import torch
import torch.nn as nn
import torch.optim as optim


class CheckpointManager:
    def __init__(
        self,
        modules: dict[str, nn.Module | optim.Optimizer],
        directory: str = "checkpoints",
    ):
        self._modules = modules
        self._directory = directory
        os.makedirs(self._directory, exist_ok=True)

    def save(self, path: str, env_step: int, gradient_step: int):
        payload = {
            **{key: module.state_dict() for key, module in self._modules.items()},
            "env_step": env_step,
            "gradient_step": gradient_step,
        }
        torch.save(payload, path)

    def load(self, path: str, device: str = "cpu"):
        checkpoint = torch.load(path, map_location=device)
        for key, module in self._modules.items():
            module.load_state_dict(checkpoint[key])
        return {
            "env_step": checkpoint["env_step"],
            "gradient_step": checkpoint["gradient_step"],
        }

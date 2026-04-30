import os

import torch
import torch.nn as nn
import torch.optim as optim


class CheckpointManager:
    def __init__(
        self,
        modules: dict[str, nn.Module | optim.Optimizer],
        directory: str = "checkpoints",
    ) -> None:
        self._modules = modules
        self._directory = directory
        os.makedirs(self._directory, exist_ok=True)

    def save(
        self,
        name: str,
        env_step: int,
        gradient_step: int,
        **extra: object,
    ) -> str:
        payload = {
            **{key: module.state_dict() for key, module in self._modules.items()},
            "env_step": env_step,
            "gradient_step": gradient_step,
        }
        filename = name.format(
            env_step=env_step,
            gradient_step=gradient_step,
            **extra,
        )
        if not filename.endswith(".pt"):
            filename += ".pt"
        path = os.path.join(self._directory, filename)
        torch.save(payload, path)
        return path

    def load(self, path: str, device: str = "cpu") -> dict[str, int]:
        if not os.path.isabs(path) and not os.path.exists(path):
            path = os.path.join(self._directory, path)
        checkpoint = torch.load(path, map_location=device)
        for key, module in self._modules.items():
            module.load_state_dict(checkpoint[key])
        return {
            "env_step": checkpoint["env_step"],
            "gradient_step": checkpoint["gradient_step"],
        }

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------


class ScalarGate(nn.Module):
    def __init__(self, eta: float, lr: float):
        super().__init__()
        self.eta = eta
        self.lr = lr

    def forward(self, _):
        return self.eta, self.lr


# ---------------------------------------------------------


def eval_memory(k, state):
    W1, b1, W2, b2 = state
    h = F.relu(F.linear(k, W1, b1))
    v_hat = F.linear(h, W2, b2)
    return v_hat


def associative_loss(state, k, v):
    v_hat = eval_memory(k, state)
    loss = torch.square(v_hat - v).sum()
    return loss


batch_eval_memory = torch.func.vmap(eval_memory)
surprise_fn = torch.func.vmap(torch.func.grad(associative_loss))


# ---------------------------------------------------------


class NeuralMemory(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int, gate: nn.Module, weight_decay: float = 0
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.gate = gate
        self.weight_decay = weight_decay

    def initial_state(self, batch_size: int, device=None):
        # (W1, b1, W2, b2) of MLP memory module
        state = (
            torch.randn(batch_size, self.hidden_dim, self.dim, device=device) * 0.01,
            torch.randn(batch_size, self.hidden_dim, device=device) * 0.01,
            torch.randn(batch_size, self.dim, self.hidden_dim, device=device) * 0.01,
            torch.randn(batch_size, self.dim, device=device) * 0.01,
        )
        momentum = tuple(torch.zeros_like(p) for p in state)
        return state, momentum

    def forward(self, k, v, state, momentum):
        k = F.normalize(k, dim=-1)
        v = F.normalize(v, dim=-1)

        surprise = tuple(s.detach() for s in surprise_fn(state, k, v))
        eta, lr = self.gate(k)

        new_momentum = tuple(eta * m + s for m, s in zip(momentum, surprise))
        new_state = tuple(
            (1 - self.weight_decay) * w - lr * m for w, m in zip(state, new_momentum)
        )
        return new_state, new_momentum

    def query(self, q, state):
        q = F.normalize(q, dim=-1)
        v_hat = batch_eval_memory(q, state)
        return v_hat

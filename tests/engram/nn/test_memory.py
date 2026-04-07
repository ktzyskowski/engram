import pytest
import torch
import torch.nn.functional as F

from engram.nn.memory import NeuralMemory, ScalarGate


B = 4  # batch size
DIM = 16
HIDDEN = 32


@pytest.fixture
def memory():
    gate = ScalarGate(eta=0.9, lr=0.1)
    return NeuralMemory(dim=DIM, hidden_dim=HIDDEN, gate=gate)


# ---------------------------------------------------------
# shapes
# ---------------------------------------------------------


def test_initial_state_shapes(memory):
    state, momentum = memory.initial_state(B)
    W1, b1, W2, b2 = state

    assert W1.shape == (B, HIDDEN, DIM)
    assert b1.shape == (B, HIDDEN)
    assert W2.shape == (B, DIM, HIDDEN)
    assert b2.shape == (B, DIM)

    for s, m in zip(state, momentum):
        assert s.shape == m.shape


def test_forward_output_shapes(memory):
    state, momentum = memory.initial_state(B)
    k = torch.randn(B, DIM)
    v = torch.randn(B, DIM)

    new_state, new_momentum = memory(k, v, state, momentum)

    for orig, updated in zip(state, new_state):
        assert orig.shape == updated.shape
    for orig, updated in zip(momentum, new_momentum):
        assert orig.shape == updated.shape


def test_query_output_shape(memory):
    state, _ = memory.initial_state(B)
    q = torch.randn(B, DIM)

    retrieved = memory.query(q, state)

    assert retrieved.shape == (B, DIM)


# ---------------------------------------------------------
# correctness
# ---------------------------------------------------------


def test_forward_mutates_state(memory):
    state, momentum = memory.initial_state(B)
    k = torch.randn(B, DIM)
    v = torch.randn(B, DIM)

    new_state, _ = memory(k, v, state, momentum)

    # At least one weight tensor must have changed
    assert any(not torch.equal(w_old, w_new) for w_old, w_new in zip(state, new_state))


def test_store_and_retrieve_converges(memory):
    """Writing the same (k, v) pair repeatedly should drive query(k) toward v."""
    state, momentum = memory.initial_state(1)

    k = torch.randn(1, DIM)
    v = torch.randn(1, DIM)

    for _ in range(200):
        state, momentum = memory(k, v, state, momentum)

    retrieved = memory.query(k, state)

    # Compare in normalized space since query normalizes q internally
    v_norm = F.normalize(v, dim=-1)
    similarity = F.cosine_similarity(retrieved, v_norm, dim=-1).item()

    assert similarity > 0.9, f"Expected cosine similarity > 0.9, got {similarity:.4f}"

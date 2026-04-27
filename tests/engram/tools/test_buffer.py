import numpy as np
import pytest
import torch

from engram.tools.buffer import ReplayBuffer


def _make_buffer(capacity=10, obs_shape=(2,), action_size=3, dtype="float32"):
    return ReplayBuffer(
        observation_shape=obs_shape,
        action_size=action_size,
        capacity=capacity,
        dtype=dtype,
    )


def _fill(buffer, n, dones=None):
    """Fill buffer with `n` transitions where obs[t] = [t, t]. Returns the timestamps written."""
    timestamps = []
    for t in range(n):
        done = bool(dones[t]) if dones is not None else False
        buffer.add(
            observation=np.array([t, t], dtype=np.float32),
            action=np.zeros(3, dtype=np.float32),
            reward=float(t),
            done=done,
        )
        timestamps.append(t)
    return timestamps


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        ReplayBuffer(observation_shape=(2,), action_size=1, capacity=0)


def test_len_before_and_after_full():
    buffer = _make_buffer(capacity=5)
    assert len(buffer) == 0
    _fill(buffer, 3)
    assert len(buffer) == 3
    _fill(buffer, 5)  # total 8, wraps once
    assert len(buffer) == 5
    assert buffer.is_full


def test_add_wraps_around():
    buffer = _make_buffer(capacity=4)
    _fill(buffer, 6)
    assert buffer.is_full
    # last write was index (6-1) % 4 = 1, so next write is index 2
    assert buffer.buffer_index == 2
    # oldest slot (index 2) holds the obs from t=5? no — t=4 wrote slot 0, t=5 wrote slot 1.
    # so slot 2 still holds t=2.
    assert buffer.observations[2][0] == 2
    assert buffer.observations[1][0] == 5  # most recent


def test_add_accepts_torch_tensors():
    buffer = _make_buffer(capacity=4)
    buffer.add(
        observation=torch.tensor([1.0, 2.0]),
        action=torch.tensor([0.0, 1.0, 0.0]),
        reward=0.5,
        done=False,
    )
    assert buffer.observations[0][0] == 1.0
    assert buffer.actions[0][1] == 1.0


def test_sample_raises_when_buffer_too_small():
    buffer = _make_buffer(capacity=10)
    _fill(buffer, 3)
    with pytest.raises(ValueError):
        buffer.sample(batch_size=1, sequence_length=5)


def test_sample_shapes_and_keys():
    buffer = _make_buffer(capacity=20, obs_shape=(2,), action_size=3)
    _fill(buffer, 15)
    batch = buffer.sample(batch_size=4, sequence_length=6)
    assert set(batch) == {"observations", "actions", "rewards", "dones"}
    assert batch["observations"].shape == (4, 6, 2)
    assert batch["actions"].shape == (4, 6, 3)
    assert batch["rewards"].shape == (4, 6)
    assert batch["dones"].shape == (4, 6)


def test_sample_not_full_returns_contiguous_timestamps():
    # observations carry a timestamp; consecutive entries must increase by 1.
    buffer = _make_buffer(capacity=20)
    _fill(buffer, 10)
    batch = buffer.sample(batch_size=3, sequence_length=4)
    obs = batch["observations"][..., 0]  # timestamps
    diffs = np.diff(obs, axis=1)
    assert np.all(diffs == 1)


def _gather_many(buffer, batch_size, sequence_length, n_calls):
    chunks = [
        buffer.sample(batch_size, sequence_length)["observations"][..., 0]
        for _ in range(n_calls)
    ]
    return np.concatenate(chunks, axis=0)


def test_sample_full_no_terminal_does_not_cross_write_head():
    # Wrap the buffer fully with no dones, then verify sequences never cross
    # the discontinuity at the write head.
    capacity = 10
    buffer = _make_buffer(capacity=capacity)
    _fill(buffer, 25)  # buffer_index = 25 % 10 = 5
    assert buffer.is_full
    assert not buffer.dones[(buffer.buffer_index - 1) % capacity]

    obs = _gather_many(buffer, batch_size=4, sequence_length=4, n_calls=50)
    diffs = np.diff(obs, axis=1)
    assert np.all(diffs == 1), "sequence crossed write head"


def test_sample_full_with_terminal_uses_full_capacity():
    # Exact wrap with done at the boundary: all start indices should be valid.
    capacity = 8
    buffer = _make_buffer(capacity=capacity)
    dones = [False] * (capacity - 1) + [True]
    _fill(buffer, capacity, dones=dones)
    assert buffer.is_full
    assert buffer.buffer_index == 0
    assert buffer.dones[(buffer.buffer_index - 1) % capacity]

    # Many samples should cover all capacity start positions.
    obs = _gather_many(buffer, batch_size=8, sequence_length=3, n_calls=100)
    starts = obs[:, 0].astype(int)
    assert set(np.unique(starts)) == set(range(capacity))


def test_sample_full_no_terminal_small_buffer_index_excludes_wrap():
    # Regression for the off-by-one: when buffer_index < seq_len - 1,
    # sequences must still avoid crossing the write head.
    capacity = 10
    buffer = _make_buffer(capacity=capacity)
    _fill(buffer, 11)  # buffer_index = 1, no dones
    assert buffer.is_full
    assert buffer.buffer_index == 1

    obs = _gather_many(buffer, batch_size=8, sequence_length=3, n_calls=50)
    diffs = np.diff(obs, axis=1)
    assert np.all(diffs == 1)


def test_sample_torch_returns_tensors():
    buffer = _make_buffer(capacity=10)
    _fill(buffer, 8)
    batch = buffer.sample_torch(batch_size=2, sequence_length=4)
    for v in batch.values():
        assert isinstance(v, torch.Tensor)
    assert batch["observations"].dtype == torch.float32
    assert batch["dones"].dtype == torch.float32  # cast through dtype kwarg

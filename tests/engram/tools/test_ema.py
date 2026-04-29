import torch

from engram.tools.ema import EMA


def test_constant_input_converges_to_constant():
    ema = EMA(decay=0.9)
    for _ in range(500):
        out = ema(torch.tensor(7.0))
    assert torch.allclose(out, torch.tensor(7.0), atol=1e-3)


def test_bias_correction_first_step_returns_input():
    # first call: avg = (1-decay)*x, correction = (1-decay), so output == x
    ema = EMA(decay=0.99)
    out = ema(torch.tensor(3.5))
    assert torch.allclose(out, torch.tensor(3.5), atol=1e-6)


def test_step_counter_increments():
    ema = EMA(decay=0.9)
    for _ in range(5):
        ema(torch.tensor(1.0))
    assert ema.get_buffer("step").item() == 5


def test_state_dict_roundtrip():
    ema = EMA(decay=0.9)
    for _ in range(10):
        ema(torch.tensor(2.0))
    state = ema.state_dict()

    ema2 = EMA(decay=0.9)
    ema2.load_state_dict(state)
    out1 = ema(torch.tensor(2.0))
    out2 = ema2(torch.tensor(2.0))
    assert torch.allclose(out1, out2)


def test_no_grad_on_output():
    ema = EMA(decay=0.9)
    x = torch.tensor(1.0, requires_grad=True)
    out = ema(x)
    assert not out.requires_grad

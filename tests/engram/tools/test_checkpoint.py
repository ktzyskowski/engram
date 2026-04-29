import torch
import torch.nn as nn
import torch.optim as optim

from engram.tools.checkpoint import CheckpointManager


def test_save_load_roundtrip(tmp_path):
    model = nn.Linear(4, 2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # take a step so optimizer state is non-empty
    loss = model(torch.randn(3, 4)).sum()
    loss.backward()
    optimizer.step()

    mgr = CheckpointManager(
        {"model": model, "optimizer": optimizer}, directory=str(tmp_path)
    )
    path = str(tmp_path / "ckpt.pt")
    mgr.save(path, env_step=42, gradient_step=100)

    # mutate the model so we can verify load actually overwrites
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

    info = mgr.load(path)
    assert info == {"env_step": 42, "gradient_step": 100}

    # weights should no longer be zero after load
    has_nonzero = any(p.abs().sum().item() > 0 for p in model.parameters())
    assert has_nonzero


def test_creates_directory(tmp_path):
    target = tmp_path / "new_subdir"
    CheckpointManager({}, directory=str(target))
    assert target.exists() and target.is_dir()


def test_payload_contains_module_keys(tmp_path):
    model = nn.Linear(2, 2)
    mgr = CheckpointManager({"model": model}, directory=str(tmp_path))
    path = str(tmp_path / "ckpt.pt")
    mgr.save(path, env_step=0, gradient_step=0)

    payload = torch.load(path, weights_only=False)
    assert "model" in payload
    assert "env_step" in payload
    assert "gradient_step" in payload

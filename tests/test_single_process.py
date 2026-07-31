from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

import mini_deepspeed as mds


def make_model() -> nn.Module:
    torch.manual_seed(13)
    return nn.Sequential(nn.Linear(3, 7), nn.GELU(), nn.Linear(7, 2))


def test_zero_stages_match_adamw_in_one_process() -> None:
    reference = make_model()
    inputs = torch.randn(5, 3)
    targets = torch.randn(5, 2)

    optimizer = torch.optim.AdamW(reference.parameters(), lr=2e-3, weight_decay=0.1)
    reference_loss = F.mse_loss(reference(inputs), targets)
    reference_loss.backward()
    optimizer.step()

    for stage in (0, 1, 2):
        model = make_model()
        engine = mds.initialize(model, {"zero_stage": stage, "lr": 2e-3, "weight_decay": 0.1})
        loss = F.mse_loss(engine(inputs), targets)
        engine.backward(loss)
        engine.step()
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_stage_reports_show_expected_ownership() -> None:
    model = make_model()
    total = sum(parameter.numel() for parameter in model.parameters())
    report0 = mds.initialize(model, {"zero_stage": 0}).report()
    report2 = mds.initialize(make_model(), {"zero_stage": 2}).report()

    assert report0.parameter_elements == total
    assert report0.gradient_elements == total
    assert report0.optimizer_state_elements == 2 * total
    assert report2.model_state_elements == report0.model_state_elements

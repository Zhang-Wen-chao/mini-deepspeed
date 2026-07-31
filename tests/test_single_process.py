from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest
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


def test_stage_two_requires_engine_backward_and_a_completed_backward() -> None:
    model = make_model()
    engine = mds.initialize(model, {"zero_stage": 2, "reduce_bucket_size": 10})

    with pytest.raises(RuntimeError, match="requires at least one engine.backward"):
        engine.step()

    inputs = torch.randn(5, 3)
    targets = torch.randn(5, 2)
    with pytest.raises(RuntimeError, match=r"engine.backward\(loss\)"):
        F.mse_loss(engine(inputs), targets).backward()


def test_stage_two_accumulated_backwards_match_one_combined_loss() -> None:
    inputs0, targets0 = torch.randn(5, 3), torch.randn(5, 2)
    inputs1, targets1 = torch.randn(5, 3), torch.randn(5, 2)

    accumulated_model = make_model()
    accumulated = mds.initialize(accumulated_model, {"zero_stage": 2, "lr": 2e-3, "weight_decay": 0.1})
    accumulated.zero_grad()
    accumulated.backward(F.mse_loss(accumulated(inputs0), targets0))
    accumulated.backward(F.mse_loss(accumulated(inputs1), targets1))
    accumulated.step()

    combined_model = make_model()
    combined = mds.initialize(combined_model, {"zero_stage": 2, "lr": 2e-3, "weight_decay": 0.1})
    combined.zero_grad()
    combined.backward(
        F.mse_loss(combined(inputs0), targets0) + F.mse_loss(combined(inputs1), targets1)
    )
    combined.step()

    for actual, expected in zip(accumulated_model.parameters(), combined_model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_stage_two_zero_grad_discards_reduced_shards() -> None:
    inputs0, targets0 = torch.randn(5, 3), torch.randn(5, 2)
    inputs1, targets1 = torch.randn(5, 3), torch.randn(5, 2)

    restarted_model = make_model()
    restarted = mds.initialize(restarted_model, {"zero_stage": 2, "lr": 2e-3})
    restarted.zero_grad()
    restarted.backward(F.mse_loss(restarted(inputs0), targets0))
    restarted.zero_grad()
    restarted.backward(F.mse_loss(restarted(inputs1), targets1))
    restarted.step()

    fresh_model = make_model()
    fresh = mds.initialize(fresh_model, {"zero_stage": 2, "lr": 2e-3})
    fresh.zero_grad()
    fresh.backward(F.mse_loss(fresh(inputs1), targets1))
    fresh.step()

    for actual, expected in zip(restarted_model.parameters(), fresh_model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_stage_two_incomplete_backward_requires_zero_grad_before_recovery() -> None:
    inputs, targets = torch.randn(5, 3), torch.randn(5, 2)
    recovered_model = make_model()
    recovered = mds.initialize(recovered_model, {"zero_stage": 2, "lr": 2e-3})
    recovered.zero_grad()
    with pytest.raises(RuntimeError, match="every trainable parameter"):
        recovered.backward(recovered_model[0].weight.sum())
    with pytest.raises(RuntimeError, match="invalidated"):
        recovered.step()
    with pytest.raises(RuntimeError, match="invalidated"):
        recovered.backward(F.mse_loss(recovered(inputs), targets))

    recovered.zero_grad()
    recovered.backward(F.mse_loss(recovered(inputs), targets))
    recovered.step()

    fresh_model = make_model()
    fresh = mds.initialize(fresh_model, {"zero_stage": 2, "lr": 2e-3})
    fresh.zero_grad()
    fresh.backward(F.mse_loss(fresh(inputs), targets))
    fresh.step()

    for actual, expected in zip(recovered_model.parameters(), fresh_model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_frozen_parameters_are_not_owned_or_updated() -> None:
    reference = make_model()
    frozen_reference = next(reference.parameters())
    frozen_reference.requires_grad_(False)
    inputs, targets = torch.randn(5, 3), torch.randn(5, 2)
    reference_optimizer = torch.optim.AdamW(
        (parameter for parameter in reference.parameters() if parameter.requires_grad), lr=2e-3, weight_decay=0.1
    )
    F.mse_loss(reference(inputs), targets).backward()
    reference_optimizer.step()

    for stage in (0, 1, 2):
        model = make_model()
        frozen = next(model.parameters())
        frozen.requires_grad_(False)
        frozen_before = frozen.detach().clone()
        engine = mds.initialize(model, {"zero_stage": stage, "lr": 2e-3, "weight_decay": 0.1})
        engine.zero_grad()
        engine.backward(F.mse_loss(engine(inputs), targets))
        engine.step()
        assert torch.equal(frozen, frozen_before)
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

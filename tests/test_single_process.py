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

    expected = torch.cat([parameter.detach().reshape(-1) for parameter in reference.parameters()])
    for stage in (0, 1, 2, 3):
        model = make_model()
        engine = mds.initialize(model, {"zero_stage": stage, "lr": 2e-3, "weight_decay": 0.1})
        loss = F.mse_loss(engine(inputs), targets)
        engine.backward(loss)
        engine.step()
        torch.testing.assert_close(engine.parameter_vector(), expected, rtol=1e-6, atol=1e-7)


def test_stage_reports_show_expected_ownership() -> None:
    model = make_model()
    total = sum(parameter.numel() for parameter in model.parameters())
    report0 = mds.initialize(model, {"zero_stage": 0}).report()
    report2 = mds.initialize(make_model(), {"zero_stage": 2}).report()
    report3 = mds.initialize(make_model(), {"zero_stage": 3}).report()

    assert report0.parameter_elements == total
    assert report0.gradient_elements == total
    assert report0.optimizer_state_elements == 2 * total
    assert report2.model_state_elements == report0.model_state_elements
    assert report3.parameter_elements == total
    assert report3.gradient_elements == total
    assert report3.optimizer_state_elements == 2 * total


def test_stage_two_requires_engine_backward_and_a_completed_backward() -> None:
    model = make_model()
    engine = mds.initialize(model, {"zero_stage": 2, "reduce_bucket_size": 10})

    with pytest.raises(RuntimeError, match="requires at least one engine.backward"):
        engine.step()

    inputs = torch.randn(5, 3)
    targets = torch.randn(5, 2)
    with pytest.raises(RuntimeError, match=r"engine.backward\(loss\)"):
        F.mse_loss(engine(inputs), targets).backward()


def test_stage_three_requires_one_engine_forward_per_backward() -> None:
    engine = mds.initialize(make_model(), {"zero_stage": 3, "reduce_bucket_size": 10})
    inputs = torch.randn(5, 3)
    targets = torch.randn(5, 2)

    with pytest.raises(RuntimeError, match="requires engine.forward"):
        engine.backward(torch.ones(()))

    loss = F.mse_loss(engine(inputs), targets)
    with pytest.raises(RuntimeError, match="only one engine.forward"):
        engine(inputs)
    engine.backward(loss)
    assert all(parameter.numel() == 0 for parameter in engine.module.parameters())
    engine.step()


def test_stage_three_abort_forward_releases_parameters_and_requires_reset() -> None:
    engine = mds.initialize(make_model(), {"zero_stage": 3})
    engine(torch.randn(5, 3))
    engine.abort_forward()
    assert all(parameter.numel() == 0 for parameter in engine.module.parameters())
    with pytest.raises(RuntimeError, match="invalidated"):
        engine(torch.randn(5, 3))
    engine.zero_grad()
    engine.backward(F.mse_loss(engine(torch.randn(5, 3)), torch.randn(5, 2)))
    engine.step()


def test_stage_three_rejects_checkpoints_while_parameters_are_sharded() -> None:
    engine = mds.initialize(make_model(), {"zero_stage": 3})
    with pytest.raises(RuntimeError, match="checkpointing is not implemented"):
        engine.state_dict()
    with pytest.raises(RuntimeError, match="checkpointing is not implemented"):
        engine.load_state_dict({})
    with pytest.raises(RuntimeError, match="checkpointing is not implemented"):
        engine.module.state_dict()
    with pytest.raises(RuntimeError, match="checkpointing is not implemented"):
        engine.module.load_state_dict({})
    # Pre-hooks are registered on every submodule, so reaching into a child
    # module directly must not bypass the rejection either.
    with pytest.raises(RuntimeError, match="checkpointing is not implemented"):
        engine.module[0].state_dict()
    with pytest.raises(RuntimeError, match="checkpointing is not implemented"):
        engine.module[1].load_state_dict({})


def test_parameter_view_is_rejected_before_zero_ownership_is_created() -> None:
    class ParameterView(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4))
            self.view = nn.Parameter(self.weight[:2])

    # Shared storage is rejected for every stage: the flat layout updates the
    # two copies independently and the final write-back would silently drop
    # one gradient contribution, unlike torch.optim.AdamW's compound in-place
    # updates.
    for stage in (0, 1, 2, 3):
        with pytest.raises(ValueError, match="Parameter views or shared storage"):
            mds.initialize(ParameterView(), {"zero_stage": stage})


def test_noncontiguous_parameters_are_accepted_and_match_adamw() -> None:
    class NonContiguous(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Full-storage transposed parameter: non-contiguous but not a view.
            self.weight = nn.Parameter(torch.randn(3, 4).t())

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs @ self.weight

    torch.manual_seed(3)
    inputs, targets = torch.randn(5, 4), torch.randn(5, 3)
    torch.manual_seed(3)
    reference = NonContiguous()
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=2e-3, weight_decay=0.1)
    F.mse_loss(reference(inputs), targets).backward()
    reference_optimizer.step()
    expected = torch.cat([parameter.detach().reshape(-1) for parameter in reference.parameters()])

    for stage in (0, 1, 2, 3):
        torch.manual_seed(3)
        model = NonContiguous()
        engine = mds.initialize(model, {"zero_stage": stage, "lr": 2e-3, "weight_decay": 0.1})
        engine.zero_grad()
        engine.backward(F.mse_loss(engine(inputs), targets))
        engine.step()
        torch.testing.assert_close(engine.parameter_vector(), expected, rtol=1e-6, atol=1e-7)


def test_ordinary_weight_tying_remains_supported() -> None:
    class TiedRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(3, 3, bias=False)
            self.second = nn.Linear(3, 3, bias=False)
            self.second.weight = self.first.weight

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.second(torch.tanh(self.first(inputs)))

    torch.manual_seed(91)
    reference = TiedRegressor()
    inputs, targets = torch.randn(5, 3), torch.randn(5, 3)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=2e-3, weight_decay=0.1)
    F.mse_loss(reference(inputs), targets).backward()
    reference_optimizer.step()
    expected = torch.cat([parameter.detach().reshape(-1) for parameter in reference.parameters()])

    for stage in (0, 1, 2, 3):
        torch.manual_seed(91)
        model = TiedRegressor()
        engine = mds.initialize(model, {"zero_stage": stage, "lr": 2e-3, "weight_decay": 0.1})
        engine.backward(F.mse_loss(engine(inputs), targets))
        engine.step()
        torch.testing.assert_close(engine.parameter_vector(), expected, rtol=1e-6, atol=1e-7)


def test_stage_three_accumulated_backwards_match_one_combined_loss() -> None:
    inputs0, targets0 = torch.randn(5, 3), torch.randn(5, 2)
    inputs1, targets1 = torch.randn(5, 3), torch.randn(5, 2)

    accumulated = mds.initialize(make_model(), {"zero_stage": 3, "lr": 2e-3, "weight_decay": 0.1})
    accumulated.zero_grad()
    accumulated.backward(F.mse_loss(accumulated(inputs0), targets0))
    accumulated.backward(F.mse_loss(accumulated(inputs1), targets1))
    accumulated.step()

    combined = mds.initialize(make_model(), {"zero_stage": 0, "lr": 2e-3, "weight_decay": 0.1})
    combined.zero_grad()
    combined.backward(
        F.mse_loss(combined(inputs0), targets0) + F.mse_loss(combined(inputs1), targets1)
    )
    combined.step()

    torch.testing.assert_close(accumulated.parameter_vector(), combined.parameter_vector(), rtol=1e-6, atol=1e-7)


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

    expected = torch.cat([parameter.detach().reshape(-1) for parameter in reference.parameters()])
    for stage in (0, 1, 2, 3):
        model = make_model()
        frozen = next(model.parameters())
        frozen.requires_grad_(False)
        frozen_before = frozen.detach().clone()
        engine = mds.initialize(model, {"zero_stage": stage, "lr": 2e-3, "weight_decay": 0.1})
        engine.zero_grad()
        engine.backward(F.mse_loss(engine(inputs), targets))
        engine.step()
        if stage != 3:
            assert torch.equal(frozen, frozen_before)
        # Frozen tensors are intentionally outside the ZeRO layout. Stage 3
        # leaves them resident while its trainable vector is sharded.
        trainable_actual = torch.cat(
            [parameter.detach().reshape(-1) for parameter in model.parameters() if parameter.requires_grad]
        ) if stage != 3 else engine.parameter_vector()
        trainable_expected = torch.cat(
            [parameter.detach().reshape(-1) for parameter in reference.parameters() if parameter.requires_grad]
        )
        torch.testing.assert_close(trainable_actual, trainable_expected, rtol=1e-6, atol=1e-7)

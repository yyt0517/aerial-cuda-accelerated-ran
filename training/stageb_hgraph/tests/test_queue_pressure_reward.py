import math

import pytest
import torch

from training.stageb_hgraph import reward_utils


def test_queue_pressure_increases_with_critical_backlog_and_service_gap():
    base = torch.zeros((2, 16))
    base[:, 0] = 3000.0
    high = base.clone()
    high[0, 12] = 6000.0
    high[0, 15] = 100.0
    mask = torch.tensor([True, True])

    assert reward_utils.compute_queue_pressure_potential(
        high,
        mask,
    ) > reward_utils.compute_queue_pressure_potential(base, mask)


def test_reducing_pressure_produces_positive_credit():
    shaped = reward_utils.compute_queue_pressure_shaping(
        before=1.0,
        after=0.5,
        gamma=0.99,
        weight=0.25,
    )

    assert shaped["queue_pressure_delta"] == pytest.approx(0.505)
    assert shaped["queue_pressure_reward"] == pytest.approx(0.12625)


def test_no_active_demand_has_zero_differentiable_pressure():
    features = torch.ones((2, 16), requires_grad=True)

    pressure = reward_utils.compute_queue_pressure_potential(
        features,
        torch.tensor([False, False]),
    )
    pressure.backward()

    assert pressure.item() == pytest.approx(0.0)
    assert features.grad is not None
    assert torch.equal(features.grad, torch.zeros_like(features.grad))


def test_negative_pressure_inputs_are_clamped_to_zero():
    features = torch.zeros((1, 16))
    features[0, 12:16] = -100.0

    pressure = reward_utils.compute_queue_pressure_potential(
        features,
        torch.tensor([True]),
    )

    assert pressure.item() == pytest.approx(0.0)


def test_topk_selects_only_the_largest_active_pressures():
    features = torch.zeros((3, 16))
    features[:, 12] = torch.tensor([0.0, 3000.0, 6000.0])

    top_one = reward_utils.compute_queue_pressure_potential(
        features,
        torch.tensor([True, True, True]),
        topk=1,
    )
    all_three = reward_utils.compute_queue_pressure_potential(
        features,
        torch.tensor([True, True, True]),
        topk=8,
    )

    assert top_one.item() == pytest.approx(2.0 * math.log(3.0))
    assert all_three.item() == pytest.approx(
        (0.0 + 2.0 * math.log(2.0) + 2.0 * math.log(3.0)) / 3.0
    )


def test_queue_pressure_has_finite_gradients():
    values = torch.zeros((2, 16))
    values[:, 12:16] = torch.tensor(
        [[3000.0, 6000.0, 1500.0, 100.0], [6000.0, 3000.0, 750.0, 20.0]]
    )
    features = values.requires_grad_()

    pressure = reward_utils.compute_queue_pressure_potential(
        features,
        torch.tensor([True, True]),
    )
    pressure.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad[:, 12:16].abs().sum().item() > 0.0

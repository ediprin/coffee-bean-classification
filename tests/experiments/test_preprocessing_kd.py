import torch

from bilinear_lmmd.engine.preprocessing_kd import (
    kd_loss_from_teacher_probabilities,
    mean_temperature_teacher_probabilities,
)


def test_teacher_ensemble_is_mean_of_temperature_probabilities():
    left = torch.tensor([[4.0, 0.0, -1.0]])
    right = torch.tensor([[-1.0, 3.0, 0.0]])
    temperature = 2.0

    observed = mean_temperature_teacher_probabilities(
        [left, right],
        temperature,
    )
    expected = torch.stack(
        [
            torch.softmax(left / temperature, dim=1),
            torch.softmax(right / temperature, dim=1),
        ]
    ).mean(0)

    assert torch.allclose(observed, expected)
    assert torch.allclose(observed.sum(dim=1), torch.ones(1))

    wrong = torch.softmax((left + right) / (2.0 * temperature), dim=1)
    assert not torch.allclose(observed, wrong)


def test_kd_probability_loss_backpropagates_to_student_only():
    student = torch.tensor(
        [[1.0, 0.0, -0.5], [0.1, 1.2, -0.2]],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[0.7, 0.2, 0.1], [0.15, 0.75, 0.10]],
    )
    labels = torch.tensor([0, 1])

    loss, parts = kd_loss_from_teacher_probabilities(
        student,
        teacher,
        labels,
        temperature=2.0,
        hard_weight=0.5,
        label_smoothing=0.1,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(parts["hard_ce"])
    assert torch.isfinite(parts["soft_kl"])
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()

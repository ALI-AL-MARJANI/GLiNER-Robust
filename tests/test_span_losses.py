"""Tests for the span-imbalance losses (span Dice, span-width weighting).

All tests run on CPU on tiny tensors and need no download.
"""

import math
from types import SimpleNamespace

import torch
import pytest
import torch.nn.functional as F

from gliner.modeling.base import BaseModel, BiEncoderSpanModel, UniEncoderSpanModel, UniEncoderTokenModel
from gliner.training.trainer import Trainer, TrainingArguments
from gliner.modeling.loss_functions import (
    span_dice_loss,
    span_width_weights,
    apply_positive_weights,
    focal_loss_with_logits,
)


class _SpanModel(BaseModel):
    """Concrete BaseModel exposing ``_span_loss`` without building an encoder."""

    def get_representations(self):
        pass

    def forward(self, x):
        pass

    def loss(self, x):
        pass


def _span_model():
    return _SpanModel(SimpleNamespace())


def _batch(B=2, L=4, K=3, C=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(B, L, K, C, generator=g)
    labels = torch.zeros(B, L, K, C)
    labels[0, 0, 0, 0] = 1.0
    labels[0, 1, 2, 1] = 1.0
    labels[1, 2, 1, 0] = 1.0
    prompts_mask = torch.ones(B, C)
    prompts_mask[1, 1] = 0  # second example has a single entity type
    span_mask = torch.ones(B, L, K)
    span_mask[1, 3] = 0  # padded word
    return scores, labels, prompts_mask, span_mask


# --------------------------------------------------------------------------- #
# focal / BCE
# --------------------------------------------------------------------------- #


def test_focal_disabled_equals_bce():
    logits = torch.randn(3, 7, 5)
    targets = (torch.rand(3, 7, 5) > 0.8).float()
    focal = focal_loss_with_logits(logits, targets, alpha=-1, gamma=0, reduction="sum")
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="sum")
    assert torch.allclose(focal, bce, atol=1e-4)


def test_span_loss_focal_matches_upstream_formula():
    """The refactored span loss must equal the upstream masked-sum computation."""
    scores, labels, prompts_mask, span_mask = _batch()
    B, C = scores.shape[0], scores.shape[-1]
    model = _span_model()

    got = model._span_loss(scores, labels, prompts_mask, span_mask, alpha=0.25, gamma=2.0, reduction="sum")

    per_elem = focal_loss_with_logits(scores.view(B, -1, C), labels.view(B, -1, C), alpha=0.25, gamma=2.0)
    per_elem = per_elem * prompts_mask.unsqueeze(1) * span_mask.view(B, -1, 1)
    assert torch.allclose(got, per_elem.sum(), atol=1e-5)


def test_span_loss_ignores_padding():
    """Changing logits at padded spans / padded entity types must not change the loss."""
    scores, labels, prompts_mask, span_mask = _batch()
    model = _span_model()
    for loss_type in ("focal", "dice"):
        base = model._span_loss(scores, labels, prompts_mask, span_mask, loss_type=loss_type)
        perturbed = scores.clone()
        perturbed[1, 3] += 50.0  # padded word
        perturbed[1, :, :, 1] -= 50.0  # padded entity type
        again = model._span_loss(perturbed, labels, prompts_mask, span_mask, loss_type=loss_type)
        assert torch.allclose(base, again, atol=1e-6), loss_type


def test_unknown_loss_type_raises():
    scores, labels, prompts_mask, span_mask = _batch()
    with pytest.raises(ValueError):
        _span_model()._span_loss(scores, labels, prompts_mask, span_mask, loss_type="tversky")


# --------------------------------------------------------------------------- #
# Dice
# --------------------------------------------------------------------------- #


def test_dice_known_value():
    # One positive (logit 0 -> p=0.5) and one negative (logit 0 -> p=0.5), alpha=0, smooth=0:
    # 1 - 2*0.5 / (0.5 + 0.5 + 1) = 0.5
    logits = torch.zeros(1, 2, 1)
    targets = torch.tensor([[[1.0], [0.0]]])
    mask = torch.ones_like(targets)
    loss = span_dice_loss(logits, targets, mask, alpha=0.0, smooth=0.0)
    assert math.isclose(loss.item(), 0.5, abs_tol=1e-6)


def test_dice_self_adjusting_known_value():
    # alpha=1: p~ = (1-p)p = 0.25 for p=0.5 -> 1 - 2*0.25 / (0.25 + 0.25 + 1) = 2/3
    logits = torch.zeros(1, 2, 1)
    targets = torch.tensor([[[1.0], [0.0]]])
    loss = span_dice_loss(logits, targets, torch.ones_like(targets), alpha=1.0, smooth=0.0)
    assert math.isclose(loss.item(), 2.0 / 3.0, abs_tol=1e-6)


def test_dice_range_and_perfect_prediction():
    targets = torch.tensor([[[1.0], [0.0], [0.0]]])
    mask = torch.ones_like(targets)
    perfect = torch.tensor([[[30.0], [-30.0], [-30.0]]])
    wrong = -perfect
    assert span_dice_loss(perfect, targets, mask, alpha=0.0, smooth=1e-6).item() < 1e-3
    assert span_dice_loss(wrong, targets, mask, alpha=0.0, smooth=1e-6).item() > 0.99


def test_dice_gradient_finite_and_signs():
    logits = torch.zeros(1, 3, 1, requires_grad=True)
    targets = torch.tensor([[[1.0], [0.0], [0.0]]])
    loss = span_dice_loss(logits, targets, torch.ones_like(targets), alpha=0.0, smooth=1.0)
    loss.backward()
    grad = logits.grad.flatten()
    assert torch.isfinite(grad).all()
    assert grad[0] < 0  # increasing the positive logit lowers the loss
    assert grad[1] > 0 and grad[2] > 0  # increasing negative logits raises it


def test_dice_all_negative_batch_is_finite():
    logits = torch.randn(2, 5, 3, requires_grad=True)
    targets = torch.zeros(2, 5, 3)
    loss = span_dice_loss(logits, targets, torch.ones_like(targets))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


# --------------------------------------------------------------------------- #
# span-width weighting
# --------------------------------------------------------------------------- #


def test_width_weights_values():
    w = span_width_weights(num_positions=6, max_width=3)
    expected = torch.tensor([1.0, 1 + math.log(2), 1 + math.log(3)] * 2)
    assert torch.allclose(w, expected)


def test_width_weights_only_scale_positives():
    losses = torch.ones(1, 3, 2)
    targets = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]])
    weights = torch.tensor([1.0, 2.0, 3.0])
    out = apply_positive_weights(losses, targets, weights)
    expected = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [1.0, 3.0]]])
    assert torch.equal(out, expected)


def test_width_weighting_changes_loss_when_long_positive_present():
    scores, labels, prompts_mask, span_mask = _batch()  # labels[0,1,2,1] is a width-3 positive
    model = _span_model()
    for loss_type in ("focal", "dice"):
        plain = model._span_loss(scores, labels, prompts_mask, span_mask, loss_type=loss_type)
        weighted = model._span_loss(
            scores, labels, prompts_mask, span_mask, loss_type=loss_type, use_span_width_weight=True
        )
        assert not torch.allclose(plain, weighted), loss_type


def test_width_weighting_is_noop_with_only_single_word_positives():
    scores, _, prompts_mask, span_mask = _batch()
    labels = torch.zeros_like(scores)
    labels[0, 0, 0, 0] = 1.0  # width 1 -> weight 1
    model = _span_model()
    plain = model._span_loss(scores, labels, prompts_mask, span_mask, alpha=0.25, gamma=2.0)
    weighted = model._span_loss(
        scores, labels, prompts_mask, span_mask, alpha=0.25, gamma=2.0, use_span_width_weight=True
    )
    assert torch.allclose(plain, weighted)


# --------------------------------------------------------------------------- #
# Training arguments / trainer wiring
# --------------------------------------------------------------------------- #


def test_training_arguments_defaults_are_backward_compatible(tmp_path):
    args = TrainingArguments(output_dir=str(tmp_path), report_to="none")
    assert args.loss_type == "focal"
    assert args.use_span_width_weight is False
    assert args.focal_loss_alpha == -1 and args.focal_loss_gamma == 0


def test_training_arguments_rejects_unknown_loss(tmp_path):
    with pytest.raises(ValueError):
        TrainingArguments(output_dir=str(tmp_path), report_to="none", loss_type="tversky")


def _fake_trainer(args, core):
    trainer = Trainer.__new__(Trainer)
    trainer.args = args
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda m: m)
    return trainer, SimpleNamespace(model=core)


@pytest.mark.parametrize("core_cls", [UniEncoderSpanModel, BiEncoderSpanModel])
def test_trainer_forwards_loss_options_to_span_models(tmp_path, core_cls):
    args = TrainingArguments(output_dir=str(tmp_path), report_to="none", loss_type="dice", dice_alpha=0.5)
    core = core_cls.__new__(core_cls)
    trainer, wrapper = _fake_trainer(args, core)
    kwargs = trainer._extra_loss_kwargs(wrapper)
    assert kwargs["loss_type"] == "dice" and kwargs["dice_alpha"] == 0.5


def test_trainer_defaults_add_no_kwargs(tmp_path):
    args = TrainingArguments(output_dir=str(tmp_path), report_to="none")
    trainer, wrapper = _fake_trainer(args, object())
    assert trainer._extra_loss_kwargs(wrapper) == {}


def test_trainer_rejects_unsupported_model(tmp_path):
    args = TrainingArguments(output_dir=str(tmp_path), report_to="none", use_span_width_weight=True)
    core = UniEncoderTokenModel.__new__(UniEncoderTokenModel)
    trainer, wrapper = _fake_trainer(args, core)
    with pytest.raises(ValueError):
        trainer._extra_loss_kwargs(wrapper)

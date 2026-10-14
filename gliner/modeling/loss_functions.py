import torch
import torch.nn.functional as F


def focal_loss_with_logits(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    prob_margin: float = 0.0,
    reduction: str = "none",
    label_smoothing: float = 0.0,
    normalize_prob: bool = True,
    ignore_index: int = -100,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute focal loss for binary classification with logits.

    Focal loss is designed to address class imbalance by down-weighting
    easy examples and focusing on hard examples. Originally proposed in
    RetinaNet (https://arxiv.org/abs/1708.02002) for dense object detection.

    The focal loss is defined as:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where p_t is the model's estimated probability for the correct class.

    Args:
        inputs (torch.Tensor): Predicted logits of arbitrary shape.
            The predictions for each example (not probabilities).
        targets (torch.Tensor): Ground truth binary labels with the same
            shape as inputs. Should contain 0 for negative class and 1
            for positive class.
        alpha (float, optional): Weighting factor in range (0, 1) to balance
            positive vs negative examples. Set to -1 to disable. The weight
            for positive examples is alpha, and for negative examples is
            (1 - alpha). Defaults to 0.25.
        gamma (float, optional): Exponent of the modulating factor (1 - p_t)
            to balance easy vs hard examples. Higher values give more weight
            to hard examples. Set to 0 to disable focal modulation.
            Defaults to 2.
        prob_margin (float, optional): Margin to subtract from predicted
            probabilities for negative examples. Can help with hard negative
            mining. Defaults to 0.
        reduction (str, optional): Specifies the reduction to apply to the
            output. Options are:
            - 'none': No reduction, return loss for each element
            - 'mean': Return the mean loss (normalized by valid elements)
            - 'sum': Return the sum of all losses
            Defaults to 'none'.
        label_smoothing (float, optional): Amount of label smoothing to apply.
            Targets become: target * (1 - label_smoothing) + 0.5 * label_smoothing.
            Value should be in range [0, 1]. Defaults to 0.0 (no smoothing).
        normalize_prob (bool, optional): If True, apply sigmoid to inputs to
            get probabilities. If False, treat inputs as probabilities directly.
            Defaults to True.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the loss gradient. Defaults to -100.
        eps (float, optional): Small epsilon value for numerical stability
            when computing logarithms. Defaults to 1e-6.

    Returns:
        torch.Tensor: Loss tensor. Shape depends on reduction parameter:
            - If reduction='none': same shape as inputs
            - If reduction='mean' or 'sum': scalar tensor

    Raises:
        ValueError: If an invalid reduction mode is specified.

    Example:
        >>> inputs = torch.randn(32, 100)  # logits
        >>> targets = torch.randint(0, 2, (32, 100)).float()
        >>> loss = focal_loss_with_logits(inputs, targets, reduction="mean")
    """
    # Create a mask to ignore specified index
    valid_mask = targets != ignore_index

    # Apply label smoothing if needed
    if label_smoothing != 0:
        with torch.no_grad():
            targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing

    # Apply sigmoid activation to inputs
    if normalize_prob:
        p = torch.sigmoid(inputs)
    else:
        p = inputs

    pm = torch.clamp(p - prob_margin, min=0.0, max=1.0)

    # Compute the binary cross-entropy loss without reduction
    pos_term = -targets * torch.log(p.clamp(min=eps))
    neg_term = -(1.0 - targets) * torch.log((1.0 - pm).clamp(min=eps))
    loss = pos_term + neg_term

    # Apply the valid mask to the loss
    loss = loss * valid_mask

    # Apply focal loss modulation if gamma is greater than 0
    if gamma > 0:
        p_t = p * targets + (1 - pm) * (1 - targets)
        loss = loss * ((1 - p_t) ** gamma)

    # Apply alpha weighting if alpha is specified
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Apply reduction method
    if reduction == "none":
        return loss
    elif reduction == "mean":
        # Normalize by the number of valid (non-ignored) elements
        return loss.sum() / valid_mask.sum()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(
            f"Invalid value for argument 'reduction': '{reduction}'. Supported reduction modes: 'none', 'mean', 'sum'"
        )


def cross_entropy_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "sum",
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute cross-entropy loss for multi-class classification.

    A wrapper around PyTorch's F.cross_entropy that reshapes inputs and
    targets for convenient use in GLiNER models. This function flattens
    batch and sequence dimensions before computing the loss.

    Args:
        inputs (torch.Tensor): Predicted logits of shape [..., num_classes].
            Typically shape [batch_size, seq_len, num_classes] or similar.
            Will be reshaped to [-1, num_classes] internally.
        targets (torch.Tensor): Ground truth class indices with shape [...].
            Should have one fewer dimension than inputs (no class dimension).
            Values should be in range [0, num_classes - 1] or equal to
            ignore_index. Will be reshaped to [-1] internally.
        reduction (str, optional): Specifies the reduction to apply to the
            output. Options are:
            - 'none': No reduction, return loss for each element
            - 'mean': Return the mean loss
            - 'sum': Return the sum of all losses
            Defaults to 'sum'.
        label_smoothing (float, optional): Amount of label smoothing to apply.
            Value should be in range [0, 1]. Defaults to 0.0 (no smoothing).
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the loss gradient. Defaults to -100.

    Returns:
        torch.Tensor: Loss tensor. Shape depends on reduction parameter:
            - If reduction='none': shape [batch_size * seq_len] (flattened)
            - If reduction='mean' or 'sum': scalar tensor

    Example:
        >>> inputs = torch.randn(8, 50, 10)  # batch=8, seq_len=50, classes=10
        >>> targets = torch.randint(0, 10, (8, 50))  # class indices
        >>> loss = cross_entropy_loss(inputs, targets, reduction="mean")
    """
    cls_size = inputs.shape[-1]
    inputs = inputs.reshape(-1, cls_size)
    targets = targets.reshape(-1)

    loss = F.cross_entropy(
        inputs, targets, ignore_index=ignore_index, label_smoothing=label_smoothing, reduction=reduction
    )

    return loss


def span_width_weights(
    num_positions: int,
    max_width: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Per-position weights ``w(k) = 1 + ln(k)`` for flattened span candidates.

    Span models enumerate candidates as ``(start, width)`` pairs flattened to
    ``start * max_width + (width - 1)``, so the width of position ``i`` is
    ``i % max_width + 1``. Single-word spans get weight 1; a 12-word span gets
    ``1 + ln(12) ≈ 3.48``.

    This is a heuristic, not a published method: long entities are rarer than
    short ones, so their positive pairs are up-weighted. It is meant to be
    applied to positive pairs only (see :func:`apply_positive_weights`),
    because long candidates are overwhelmingly negatives.

    Args:
        num_positions (int): Number of flattened span positions (``L * K``).
        max_width (int): Maximum span width ``K`` used by the model.
        device (torch.device, optional): Device of the returned tensor.
        dtype (torch.dtype, optional): Dtype of the returned tensor.

    Returns:
        torch.Tensor: Weights of shape ``(num_positions,)``.
    """
    widths = torch.arange(num_positions, device=device) % max_width + 1
    return 1.0 + torch.log(widths.to(dtype))


def apply_positive_weights(losses: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Scale the loss of positive pairs by ``weights``; negatives keep weight 1.

    Args:
        losses (torch.Tensor): Element-wise losses of shape ``(B, N, C)``.
        targets (torch.Tensor): Binary targets with the same shape.
        weights (torch.Tensor): Per-position weights of shape ``(N,)``.

    Returns:
        torch.Tensor: Re-weighted losses with the same shape as ``losses``.
    """
    w = weights.view(1, -1, 1)
    return losses * (1.0 + targets * (w - 1.0))


def span_dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    alpha: float = 1.0,
    smooth: float = 1.0,
    square_denominator: bool = False,
    pos_weights: torch.Tensor = None,
) -> torch.Tensor:
    """Self-adjusting Dice loss over all valid (span, type) pairs of a batch.

    Follows the set-level formulation of the official implementation of
    Li et al., "Dice Loss for Data-imbalanced NLP Tasks" (ACL 2020,
    https://github.com/ShannonAI/dice_loss_for_NLP): probabilities are
    modulated by ``(1 - p)^alpha`` and a single Dice coefficient is computed
    over the whole batch::

        p~   = (1 - p)^alpha * p
        loss = 1 - (2 * sum(w * p~ * y) + smooth) / (sum(w * p~) + sum(w * y) + smooth)

    where ``w`` is 1 for every pair unless ``pos_weights`` is given, in which
    case positive pairs are weighted by their span position weight.

    Unlike BCE/focal with ``reduction="sum"``, the result lies in ``[0, 1]``
    and does not scale with the number of candidates.

    Args:
        inputs (torch.Tensor): Logits of shape ``(B, N, C)``.
        targets (torch.Tensor): Binary targets of shape ``(B, N, C)``.
        valid_mask (torch.Tensor): 1 for pairs that contribute (real span and
            real entity type), 0 for padding. Shape ``(B, N, C)``.
        alpha (float, optional): Self-adjusting exponent. ``0`` gives the plain
            soft Dice loss. Defaults to 1.0.
        smooth (float, optional): Additive smoothing (``gamma`` in the paper).
            Defaults to 1.0.
        square_denominator (bool, optional): Use squared terms in the
            denominator (V-Net variant). Defaults to False.
        pos_weights (torch.Tensor, optional): Per-position weights of shape
            ``(N,)`` applied to positive pairs. Defaults to None.

    Returns:
        torch.Tensor: Scalar loss.
    """
    p = torch.sigmoid(inputs)
    y = targets.to(p.dtype)
    m = valid_mask.to(p.dtype)

    if pos_weights is not None:
        m = m * (1.0 + y * (pos_weights.view(1, -1, 1).to(p.dtype) - 1.0))

    p_mod = ((1.0 - p) ** alpha) * p if alpha > 0 else p

    intersection = (m * p_mod * y).sum()
    if square_denominator:
        denominator = (m * p_mod * p_mod).sum() + (m * y * y).sum()
    else:
        denominator = (m * p_mod).sum() + (m * y).sum()

    return 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)

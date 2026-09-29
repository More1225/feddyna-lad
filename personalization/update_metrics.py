from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch


EPSILON = 1e-12
NEUTRAL_SCORE = 0.5


def get_floating_parameter_keys(
    global_state_dict: Mapping[str, torch.Tensor],
) -> Tuple[str, ...]:
    """Return a stable key order for floating-point model state entries."""
    return tuple(
        key
        for key, value in global_state_dict.items()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    )


def flatten_client_update_and_participation(
    client_state_dict: Mapping[str, torch.Tensor],
    global_state_dict: Mapping[str, torch.Tensor],
    client_mask: Optional[Mapping[str, torch.Tensor]],
    aggregation_scope: str,
    parameter_keys: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten a client update and its aggregation participation in one key order.

    Participation follows the server aggregation semantics:
    - all: every floating-point position participates;
    - masked_only: mask=0 participates and mask=1 remains local.
    """
    if aggregation_scope not in {"all", "masked_only"}:
        raise ValueError(f"Unsupported aggregation_scope: {aggregation_scope}")

    update_parts = []
    participation_parts = []

    for key in parameter_keys:
        if key not in client_state_dict or key not in global_state_dict:
            raise KeyError(f"Missing model state entry for update metric: {key}")

        client_value = client_state_dict[key].detach().float().cpu()
        global_value = global_state_dict[key].detach().float().cpu()
        if client_value.shape != global_value.shape:
            raise ValueError(
                f"Client/global shape mismatch for {key}: "
                f"{tuple(client_value.shape)} != {tuple(global_value.shape)}"
            )

        update_parts.append((client_value - global_value).reshape(-1))

        if aggregation_scope == "all":
            participation = torch.ones_like(client_value, dtype=torch.bool)
        elif client_mask is None or key not in client_mask:
            # This mirrors the client-state accumulator, which treats a missing
            # mask entry as fully shared.
            participation = torch.ones_like(client_value, dtype=torch.bool)
        else:
            mask_value = client_mask[key].detach().cpu()
            if mask_value.shape != client_value.shape:
                raise ValueError(
                    f"Mask/client shape mismatch for {key}: "
                    f"{tuple(mask_value.shape)} != {tuple(client_value.shape)}"
                )
            participation = mask_value == 0

        participation_parts.append(participation.reshape(-1))

    if not update_parts:
        raise ValueError("No floating-point model entries available for update metrics")

    return (
        torch.cat(update_parts).to(dtype=torch.float32),
        torch.cat(participation_parts).to(dtype=torch.bool),
    )


def build_scope_aware_client_updates(
    client_ids: Iterable[int],
    client_state_dicts: Mapping[int, Mapping[str, torch.Tensor]],
    global_state_dict: Mapping[str, torch.Tensor],
    client_masks: Optional[Mapping[int, Mapping[str, torch.Tensor]]],
    aggregation_scope: str,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Tuple[str, ...]]:
    """Build aligned update and participation vectors for all clients."""
    parameter_keys = get_floating_parameter_keys(global_state_dict)
    updates: Dict[int, torch.Tensor] = {}
    participation: Dict[int, torch.Tensor] = {}

    for cid in client_ids:
        client_mask = client_masks.get(cid) if client_masks is not None else None
        update, participate = flatten_client_update_and_participation(
            client_state_dict=client_state_dicts[cid],
            global_state_dict=global_state_dict,
            client_mask=client_mask,
            aggregation_scope=aggregation_scope,
            parameter_keys=parameter_keys,
        )
        updates[cid] = update
        participation[cid] = participate

    return updates, participation, parameter_keys


def _mapped_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    epsilon: float = EPSILON,
) -> float:
    """Map cosine similarity from [-1, 1] to [0, 1]."""
    if left.numel() == 0 or right.numel() == 0:
        return NEUTRAL_SCORE

    left_norm = float(torch.linalg.vector_norm(left).item())
    right_norm = float(torch.linalg.vector_norm(right).item())
    if left_norm <= epsilon or right_norm <= epsilon:
        return NEUTRAL_SCORE

    cosine = float(torch.dot(left, right).item()) / (
        left_norm * right_norm + epsilon
    )
    cosine = max(-1.0, min(1.0, cosine))
    return 0.5 * (cosine + 1.0)


def _shrink_to_neutral(score: float, coverage: float) -> float:
    coverage = max(0.0, min(1.0, float(coverage)))
    return float(NEUTRAL_SCORE + coverage * (float(score) - NEUTRAL_SCORE))


def compute_scope_aware_update_metrics(
    client_ids: Sequence[int],
    client_updates: Mapping[int, torch.Tensor],
    participation: Mapping[int, torch.Tensor],
    previous_updates: Optional[Mapping[int, torch.Tensor]] = None,
    previous_participation: Optional[Mapping[int, torch.Tensor]] = None,
    epsilon: float = EPSILON,
) -> Dict[str, Dict[int, float]]:
    """
    Compute leave-one-out update alignment and temporal update stability.

    The leave-one-out consensus gives every participating client equal weight
    at each parameter position. Client sample count is intentionally excluded
    here because statistical support is represented separately by the
    log_sample_count entropy factor. Sparse overlap is shrunk continuously
    toward the neutral score 0.5.
    """
    ids = list(client_ids)
    if not ids:
        return {
            "update_alignment": {},
            "update_stability": {},
            "alignment_coverage": {},
            "stability_coverage": {},
        }

    vector_length = int(client_updates[ids[0]].numel())
    update_sum = torch.zeros(vector_length, dtype=torch.float64)
    participation_count = torch.zeros(vector_length, dtype=torch.float64)

    for cid in ids:
        update = client_updates[cid].detach().cpu().to(dtype=torch.float64)
        part = participation[cid].detach().cpu().to(dtype=torch.bool)
        if update.numel() != vector_length or part.numel() != vector_length:
            raise ValueError("All client update and participation vectors must align")

        part_float = part.to(dtype=torch.float64)
        update_sum += part_float * update
        participation_count += part_float

    alignments: Dict[int, float] = {}
    stabilities: Dict[int, float] = {}
    alignment_coverage: Dict[int, float] = {}
    stability_coverage: Dict[int, float] = {}

    for cid in ids:
        current_update = client_updates[cid].detach().cpu().to(dtype=torch.float64)
        current_part = participation[cid].detach().cpu().to(dtype=torch.bool)
        current_part_float = current_part.to(dtype=torch.float64)

        other_update_sum = update_sum - current_part_float * current_update
        other_participation_count = participation_count - current_part_float
        alignment_valid = current_part & (other_participation_count > epsilon)

        own_shared_count = max(int(current_part.sum().item()), 1)
        valid_alignment_count = int(alignment_valid.sum().item())
        alignment_cov = valid_alignment_count / own_shared_count
        alignment_coverage[cid] = float(alignment_cov)

        if valid_alignment_count == 0:
            alignment_score = NEUTRAL_SCORE
        else:
            consensus = (
                other_update_sum[alignment_valid]
                / other_participation_count[alignment_valid].clamp_min(epsilon)
            )
            raw_alignment = _mapped_cosine(
                current_update[alignment_valid],
                consensus,
                epsilon=epsilon,
            )
            alignment_score = _shrink_to_neutral(raw_alignment, alignment_cov)
        alignments[cid] = float(alignment_score)

        has_history = (
            previous_updates is not None
            and previous_participation is not None
            and cid in previous_updates
            and cid in previous_participation
        )
        if not has_history:
            stabilities[cid] = NEUTRAL_SCORE
            stability_coverage[cid] = 0.0
            continue

        previous_update = (
            previous_updates[cid].detach().cpu().to(dtype=torch.float64)
        )
        previous_part = (
            previous_participation[cid].detach().cpu().to(dtype=torch.bool)
        )
        if (
            previous_update.numel() != vector_length
            or previous_part.numel() != vector_length
        ):
            stabilities[cid] = NEUTRAL_SCORE
            stability_coverage[cid] = 0.0
            continue

        stability_valid = current_part & previous_part
        valid_stability_count = int(stability_valid.sum().item())
        stability_cov = valid_stability_count / own_shared_count
        stability_coverage[cid] = float(stability_cov)

        if valid_stability_count == 0:
            stability_score = NEUTRAL_SCORE
        else:
            raw_stability = _mapped_cosine(
                current_update[stability_valid],
                previous_update[stability_valid],
                epsilon=epsilon,
            )
            stability_score = _shrink_to_neutral(raw_stability, stability_cov)
        stabilities[cid] = float(stability_score)

    return {
        "update_alignment": alignments,
        "update_stability": stabilities,
        "alignment_coverage": alignment_coverage,
        "stability_coverage": stability_coverage,
    }

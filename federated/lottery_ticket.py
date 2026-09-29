import copy
import numpy as np
import torch
from collections import OrderedDict

from utils.debug_tools import log


def init_mask_zeros(model) -> OrderedDict:
    """
    Initialize an all-zero mask:
    mask=0 -> shared parameters
    mask=1 -> personalized parameters
    An all-zero mask means all parameters are initially shared globally.
    """
    mask = OrderedDict()
    for name, param in model.state_dict().items():
        if name.endswith(".weight") or name.endswith(".bias"):
            mask[name] = torch.zeros_like(param, dtype=torch.float32)
    return mask


def _compute_delta_tensor(current_state_dict: OrderedDict, prev_state_dict: OrderedDict):
    delta_dict = OrderedDict()
    for name, param in current_state_dict.items():
        if not (name.endswith(".weight") or name.endswith(".bias")):
            continue
        delta_dict[name] = torch.abs(param.detach().cpu() - prev_state_dict[name].detach().cpu())
    return delta_dict


def _build_mask_from_global_ratio(
    current_state_dict: OrderedDict,
    prev_state_dict: OrderedDict,
    current_mask: OrderedDict,
    global_ratio: float,
    args=None,
    enforce_cumulative_personal_bounds: bool = False,
) -> OrderedDict:
    """
    Reassign the mask according to global_ratio:
    - Parameters with smaller changes -> shared (mask=0)
    - Parameters with larger changes -> personalized (mask=1)

    global_ratio is the fraction of currently mask=0 parameters retained in this round.

    KDARS uses enforce_cumulative_personal_bounds=True:
    - Retain multiplicative shrinkage across rounds;
    - personal_min_ratio / personal_max_ratio constrain the cumulative personalized fraction relative to all eligible
      parameters;
    - Parameters that have been personalized do not become shared again.
    """
    global_ratio = float(np.clip(global_ratio, 0.0, 1.0))
    new_mask = copy.deepcopy(current_mask)

    delta_dict = _compute_delta_tensor(current_state_dict, prev_state_dict)

    # Collect deltas for currently mask=0 parameters and count total and cumulative personalized parameters.
    global_delta_values = []
    global_positions = []  # Store (name, flat_index) pairs to map back to original positions.
    total_eligible_params = 0
    current_local_params = 0

    for name, delta_tensor in delta_dict.items():
        if name not in current_mask:
            continue
        mask_tensor = current_mask[name]
        total_eligible_params += int(mask_tensor.numel())
        current_local_params += int(torch.count_nonzero(mask_tensor).item())
        # Find positions currently marked mask=0.
        global_indices = (mask_tensor == 0).cpu().numpy()
        if not np.any(global_indices):
            continue

        delta_np = delta_tensor.numpy()
        # Get delta values at shared positions.
        global_deltas = delta_np[global_indices]
        # Record position metadata.
        flat_indices = np.where(global_indices.flatten())[0]
        for idx, val in zip(flat_indices, global_deltas.flatten()):
            global_delta_values.append(val)
            global_positions.append((name, idx))

    if len(global_delta_values) == 0:
        return new_mask

    current_global_params = len(global_delta_values)
    # Default semantics: retain a global_ratio fraction of the currently shared parameters this round.
    global_count = int(round(global_ratio * current_global_params))
    global_count = max(0, min(global_count, current_global_params))
    exact_local_positions = None

    if enforce_cumulative_personal_bounds:
        personal_min_ratio = float(
            np.clip(getattr(args, "personal_min_ratio", 0.2), 0.0, 1.0)
        )
        personal_max_ratio = float(
            np.clip(getattr(args, "personal_max_ratio", 0.8), 0.0, 1.0)
        )
        if personal_min_ratio > personal_max_ratio:
            raise ValueError(
                "personal_min_ratio cannot be greater than "
                "personal_max_ratio"
            )

        # KDARS provides an incremental per-round ratio. First apply multiplicative shrinkage to obtain a candidate cumulative ratio,
        # then constrain the final cumulative number of personalized parameters to [min, max] of all parameters.
        raw_new_local_count = current_global_params - global_count
        candidate_local_count = current_local_params + raw_new_local_count
        min_local_count = int(
            np.ceil(personal_min_ratio * total_eligible_params - 1e-12)
        )
        max_local_count = int(
            np.floor(personal_max_ratio * total_eligible_params + 1e-12)
        )

        if current_local_params > max_local_count:
            # For compatibility with previously saved masks, keep personalization one-way: do not restore parameters or shrink further.
            target_local_count = current_local_params
        else:
            target_local_count = int(
                np.clip(
                    candidate_local_count,
                    min_local_count,
                    max_local_count,
                )
            )
            target_local_count = max(
                current_local_params,
                target_local_count,
            )

        new_local_count = target_local_count - current_local_params
        global_count = current_global_params - new_local_count
        global_count = max(0, min(global_count, current_global_params))

        # Select exactly the new_local_count positions with the largest changes to avoid tied delta values causing
        # the cumulative ratio to exceed its bounds.
        if new_local_count > 0:
            order = np.argsort(
                np.asarray(global_delta_values, dtype=np.float64),
                kind="stable",
            )
            exact_local_positions = {
                global_positions[int(index)]
                for index in order[-new_local_count:]
            }
        else:
            exact_local_positions = set()

        current_personal_ratio = (
            current_local_params / total_eligible_params
            if total_eligible_params > 0
            else 0.0
        )
        candidate_personal_ratio = (
            candidate_local_count / total_eligible_params
            if total_eligible_params > 0
            else 0.0
        )
        target_personal_ratio = (
            target_local_count / total_eligible_params
            if total_eligible_params > 0
            else 0.0
        )
        effective_step_personal_ratio = (
            new_local_count / current_global_params
            if current_global_params > 0
            else 0.0
        )
        log(
            "[KDARS] cumulative personalization: "
            f"before={current_personal_ratio:.4f}, "
            f"candidate={candidate_personal_ratio:.4f}, "
            f"target={target_personal_ratio:.4f}, "
            f"effective_step={effective_step_personal_ratio:.4f}",
            1,
            args,
        )

    if global_count == 0:
        threshold = -np.inf
    elif global_count == current_global_params:
        threshold = np.inf
    else:
        sorted_vals = np.sort(global_delta_values)
        threshold = sorted_vals[global_count - 1]

    # Initialize the new mask as a deep copy of the current mask.
    new_mask = copy.deepcopy(current_mask)

    # Update only positions selected for pruning.
    # Group positions by parameter name for batched updates.
    updates_by_name = {}
    for (name, flat_idx), delta_val in zip(global_positions, global_delta_values):
        if name not in updates_by_name:
            updates_by_name[name] = []
        updates_by_name[name].append((flat_idx, delta_val))

    for name, updates in updates_by_name.items():
        mask_tensor = new_mask[name]
        shape = mask_tensor.shape
        flat_mask = mask_tensor.flatten().cpu().numpy()

        for flat_idx, delta_val in updates:
            # Use exact positions when KDARS cumulative bounds are active; preserve threshold semantics for other methods.
            should_be_local = (
                (name, flat_idx) in exact_local_positions
                if exact_local_positions is not None
                else delta_val > threshold
            )
            if should_be_local:
                flat_mask[flat_idx] = 1.0

        # Reshape to the original dimensions.
        new_mask[name] = torch.from_numpy(flat_mask.reshape(shape)).to(current_mask[name].device)

    total_global = 0
    total_local = 0
    total = 0
    for name, mask_tensor in new_mask.items():
        if not (name.endswith(".weight") or name.endswith(".bias")):
            continue
        local_num = int(mask_tensor.sum().item())
        global_num = mask_tensor.numel() - local_num
        total_local += local_num
        total_global += global_num
        total += mask_tensor.numel()

    if total > 0:
        msg = (
            f"  Total: shared={total_global}/{total} ({total_global/total:.2%}), "
            f"personalized={total_local}/{total} ({total_local/total:.2%})"
        )
        if args is not None:
            log(msg, 1, args)
        else:
            print(msg)

    return new_mask


def ratio_based_update(
    current_state_dict: OrderedDict,
    prev_state_dict: OrderedDict,
    current_mask: OrderedDict,
    global_ratio: float,
    args=None,
) -> OrderedDict:
    """Apply one KDARS incremental global-retention ratio cumulatively."""
    return _build_mask_from_global_ratio(
        current_state_dict=current_state_dict,
        prev_state_dict=prev_state_dict,
        current_mask=current_mask,
        global_ratio=global_ratio,
        args=args,
        enforce_cumulative_personal_bounds=True,
    )

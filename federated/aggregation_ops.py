from collections import OrderedDict
import copy
import torch


def init_server_like(model_state_dict):
    """
    Initialize server accumulators with the same structure as the model parameters.

    server_state:
        Accumulates weighted client parameters.
    server_mask:
        Accumulates the effective participation weight (denominator) at each parameter position.
    """
    server_state = OrderedDict()
    server_mask = OrderedDict()
    for key, value in model_state_dict.items():
        server_state[key] = torch.zeros_like(value)
        server_mask[key] = torch.zeros_like(value, dtype=torch.float32, device=value.device)
    return server_state, server_mask


def _build_participation_tensor(mask_tensor, aggregation_scope):
    """
    Build per-parameter participation coefficients from the aggregation scope.

    Semantics:
    - aggregation_scope == "all":
        All parameters participate in aggregation with coefficient 1.
    - aggregation_scope == "masked_only":
        Only mask=0 parameters participate; mask=1 parameters are excluded.
    """
    if aggregation_scope == "all":
        return torch.ones_like(mask_tensor, dtype=torch.float32)
    if aggregation_scope == "masked_only":
        return (mask_tensor == 0).float()
    raise ValueError(f"Unsupported aggregation scope: {aggregation_scope}")


def add_masks_weighted(
    server_mask,
    client_mask,
    weight,
    aggregation_scope="masked_only",
):
    """
    Accumulate the effective participation weight at each parameter position.

    Notes:
    - weight is the adaptive client-level aggregation weight.
    - aggregation_scope determines whether a parameter position participates in aggregation.
    """
    from utils.debug_tools import log

    if len(client_mask) > 0:
        first_key = next(iter(client_mask))
        mask_tensor = client_mask[first_key].float()
        zero_ratio = (mask_tensor == 0).sum().item() / mask_tensor.numel()
        log(
            f"[Aggregation] Mask zero ratio: {zero_ratio:.4f}, "
            f"scope={aggregation_scope}, weight={weight:.6f}",
            level=2,
        )

    for key in server_mask.keys():
        if key not in client_mask:
            continue

        target_device = server_mask[key].device
        cur = client_mask[key].float().to(target_device)
        cur = _build_participation_tensor(cur, aggregation_scope)

        server_mask[key] += weight * cur
    return server_mask


def add_server_weights_weighted(
    server_state,
    client_state,
    client_mask,
    weight,
    aggregation_scope="masked_only",
):
    """
    Accumulate client parameters on the server.

    Notes:
    - weight is the adaptive client-level aggregation weight.
    - aggregation_scope determines whether a parameter position participates in aggregation.
    - If a key is absent from client_mask, the entire tensor participates by default.
    """
    from utils.debug_tools import log

    if len(client_state) > 0:
        first_key = next(iter(client_state))
        param_norm = client_state[first_key].float().norm().item()
        log(f"[Aggregation] Client param norm: {param_norm:.6f}, weight={weight:.6f}", level=2)

    for key in server_state.keys():
        target_device = server_state[key].device
        client_tensor = client_state[key].to(target_device)

        if key not in client_mask:
            server_state[key] += weight * client_tensor
            continue

        cur = client_mask[key].float().to(target_device)
        cur = _build_participation_tensor(cur, aggregation_scope)

        server_state[key] += weight * cur * client_tensor
    return server_state


def div_server_weights(
    server_state,
    server_mask,
    prev_global_state=None,
):
    """
    Normalize accumulated server parameters elementwise by their effective participation weights.

    Args:
        server_state:       Accumulated weighted parameter sum
        server_mask:        Accumulated participation weights (denominator）
        prev_global_state:  Previous global-model parameters, used to fill positions with denominator=0:
                            when no client contributes to a parameter position (all clients have
                            mask=1 at that position), use the previous global parameter instead of leaving zero.

    Returns:
        out: Normalized global parameters as an OrderedDict
        zero_denom_ratio: float, Fraction of all parameters with denominator=0
    """
    out = copy.deepcopy(server_state)
    total_params = 0
    zero_denom_params = 0

    for key in out.keys():
        target_device = out[key].device
        mask_tensor = server_mask[key].to(target_device)

        valid = mask_tensor != 0
        # Count parameter positions with denominator=0
        num_total = mask_tensor.numel()
        num_zero_denom = (~valid).sum().item()
        # Exclude non-floating-point parameters (such as integer num_batches_tracked) from statistics.
        if out[key].is_floating_point():
            total_params += num_total
            zero_denom_params += num_zero_denom

        # For valid positions, divide by the denominator.
        out[key][valid] = out[key][valid] / mask_tensor[valid]

        # For positions with denominator=0, use the previous global parameter instead of leaving zero.
        # Note: prev_global_state is read-only; it is not deep-copied or modified in place.
        if num_zero_denom > 0 and prev_global_state is not None and key in prev_global_state:
            prev_val = prev_global_state[key].to(target_device)
            out[key][~valid] = prev_val[~valid]

    zero_denom_ratio = zero_denom_params / total_params if total_params > 0 else 0.0
    return out, zero_denom_ratio


# ===== Fallback equal-weight aggregation interfaces (their semantics should remain aligned with the main path if used independently). =====

def aggregate_model(params_list, mask_list, aggregation_scope="masked_only"):
    """
    Fallback aggregation interface with semantics aligned to the main path.

    Notes:
    - This implements equal-weight client aggregation only.
    - aggregation_scope supports only "masked_only" and "all".
    """
    if aggregation_scope == "masked_only":
        return aggregate_masked_only(params_list, mask_list)
    if aggregation_scope == "all":
        return aggregate_all(params_list)
    raise ValueError(f"Unsupported aggregation scope: {aggregation_scope}")


def aggregate_masked_only(params_list, mask_list):
    """
    Equal-weight variant: aggregate only mask=0 parameters.
    If no client contributes to a parameter position, that position remains zero.
    """
    aggregated_params = OrderedDict()
    for key in params_list[0].keys():
        param_sum = torch.zeros_like(params_list[0][key])
        weight_sum = torch.zeros_like(params_list[0][key], dtype=torch.float32)

        for params, mask in zip(params_list, mask_list):
            if key not in mask:
                param_sum += params[key]
                weight_sum += torch.ones_like(params[key], dtype=torch.float32)
                continue

            participation = (mask[key] == 0).float().to(params[key].device)
            param_sum += params[key] * participation
            weight_sum += participation

        aggregated = torch.zeros_like(param_sum)
        valid = weight_sum != 0
        aggregated[valid] = param_sum[valid] / weight_sum[valid]
        aggregated_params[key] = aggregated

    return aggregated_params


def aggregate_all(params_list):
    """
    Equal-weight aggregation over all parameters.
    """
    aggregated_params = OrderedDict()
    for key in params_list[0].keys():
        param_sum = sum([p[key] for p in params_list])
        aggregated_params[key] = param_sum / len(params_list)
    return aggregated_params

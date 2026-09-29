import torch

GLOBAL_MASK_VALUE = 0
LOCAL_MASK_VALUE = 1


def is_global(mask_tensor: torch.Tensor) -> torch.Tensor:
    return mask_tensor == GLOBAL_MASK_VALUE


def is_local(mask_tensor: torch.Tensor) -> torch.Tensor:
    return mask_tensor == LOCAL_MASK_VALUE


def global_mask(mask_tensor: torch.Tensor) -> torch.Tensor:
    return (mask_tensor == GLOBAL_MASK_VALUE).to(mask_tensor.dtype)


def local_mask(mask_tensor: torch.Tensor) -> torch.Tensor:
    return (mask_tensor == LOCAL_MASK_VALUE).to(mask_tensor.dtype)


def count_mask(mask_tensor: torch.Tensor) -> dict[str, int]:
    total = mask_tensor.numel()
    global_count = int((mask_tensor == GLOBAL_MASK_VALUE).sum().item())
    local_count = int((mask_tensor == LOCAL_MASK_VALUE).sum().item())
    return {
        "total": total,
        "global": global_count,
        "local": local_count,
    }
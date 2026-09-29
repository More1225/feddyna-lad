import os
import math
import torch
import matplotlib.pyplot as plt


def log(msg: str, level: int = 1, args=None):
    if args is None:
        print(msg)
        return
    if getattr(args, "verbose", 1) >= level:
        print(msg)


def print_section(title: str, char: str = "="):
    line = char * 80
    print(f"\n{line}\n{title}\n{line}")


def compute_grad_norm(model: torch.nn.Module) -> float:
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2).item()
            total_norm_sq += param_norm ** 2
    return math.sqrt(total_norm_sq)


def summarize_state_dict(state_dict, topk: int = 5):
    rows = []
    for k, v in state_dict.items():
        rows.append({
            "name": k,
            "mean": float(v.float().mean().item()),
            "std": float(v.float().std().item()),
            "norm": float(v.float().norm().item()),
            "shape": tuple(v.shape),
        })
    rows = sorted(rows, key=lambda x: x["norm"], reverse=True)
    return rows[:topk]


def print_state_summary(state_dict, title="State Summary", topk: int = 5):
    print(f"\n[DEBUG] {title}")
    for item in summarize_state_dict(state_dict, topk=topk):
        print(
            f"  {item['name']}: shape={item['shape']}, "
            f"mean={item['mean']:.6f}, std={item['std']:.6f}, norm={item['norm']:.6f}"
        )


def save_client_training_plot(
    round_num: int,
    client_id: int,
    epoch_losses,
    epoch_accs,
    grad_norms,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f"client_{client_id}_round_{round_num}.png")

    plt.figure(figsize=(10, 6))
    epochs = list(range(1, len(epoch_losses) + 1))

    plt.plot(epochs, epoch_losses, label="Train Loss")
    if len(epoch_accs) == len(epoch_losses):
        plt.plot(epochs, epoch_accs, label="Train Accuracy")
    if len(grad_norms) == len(epoch_losses):
        plt.plot(epochs, grad_norms, label="Grad Norm")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"Client {client_id} - Round {round_num}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path
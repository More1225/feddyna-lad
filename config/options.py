import argparse
import json
import os
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Federated LightLog")

    # ========= Basic =========
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exp_name", type=str, default="fed_lightlog_exp")
    parser.add_argument("--save_dir", type=str, default="./save")
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Optional exact output directory for one run; used by reproducible batch scripts.",
    )
    parser.add_argument(
        "--scenario_tag",
        type=str,
        default="",
        help="Optional scenario label recorded in result artifacts.",
    )
    parser.add_argument(
        "--method_tag",
        type=str,
        default="",
        help="Optional method label recorded in result artifacts.",
    )

    # ========= Data =========
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset name, commonly hdfs, bgl, thunderbird, or tb. Alias support depends on the upstream data loader.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Required path to a user-supplied unified federated NPZ file.",
    )
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--input_dim", type=int, default=300)
    parser.add_argument("--local_bs", type=int, default=32)

    # ========= Model =========
    parser.add_argument("--model", type=str, default="lightlog")
    parser.add_argument("--compress_dim", type=int, default=96)
    parser.add_argument("--filters_list", type=int, nargs="+", default=[32, 16, 8])
    parser.add_argument("--dropout", type=float, default=0.3)

    # ========= Local training =========
    parser.add_argument(
        "--loss_mode",
        type=str,
        choices=["ce", "weighted_ce", "focal"],
        default="weighted_ce",
    )
    parser.add_argument(
        "--use_auto_class_weight",
        action="store_true",
        default=True,
        help="Whether to compute class weights automatically (True: derive them from class distribution; False: use fixed weights).",
    )
    parser.add_argument(
        "--pos_weight_scale",
        type=float,
        default=1.0,
        help="Positive-class weight scaling factor for weighted_ce and focal losses.",
    )
    parser.add_argument(
        "--max_pos_weight",
        type=float,
        default=10.0,
        help="Maximum positive-class weight for weighted_ce and focal losses.",
    )
    parser.add_argument(
        "--fixed_pos_weight",
        type=float,
        default=8.0,
        help="Fixed positive-class weight used when use_auto_class_weight=False.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Local-training early-stopping patience, measured in epochs.",
    )
    parser.add_argument(
        "--early_stopping_delta",
        type=float,
        default=0.0,
        help="Minimum validation-metric improvement; smaller changes count as no improvement.",
    )
    parser.add_argument(
        "--save_best_local_model",
        action="store_true",
        default=False,
        help=(
            "Whether to save each client's local model at its best validation F1."
            "True: save the best-F1 model (current behavior); False: save the final trained model."
            "Set to False to compare the final effects of different aggregation strategies."
        ),
    )
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--clipgradnorm", type=float, default=1.0)
    parser.add_argument("--local_epochs", type=int, default=5)

    # ========= Federated learning =========
    parser.add_argument("--com_rounds", type=int, default=6)

    # ========= Federated method selection =========
    parser.add_argument(
        "--fl_method",
        type=str,
        default="fedavg",
        choices=["fedavg", "fedprox", "ditto", "fedrep", "fedpac", "fedala", "fedapa", "fedas", "fedltn", "fedselect", "local"],
        help="Federated method: fedavg=standard federated averaging, fedprox=FedProx, ditto=Ditto, "
             "fedrep=FedRep, fedpac=FedPAC, fedala=FedALA, fedapa=FedAPA, "
             "fedas=FedAS, fedltn=FedLTN, fedselect=standalone FedSelect baseline, local=independent local training (no federated aggregation)"
    )

    # ========= FedProx options =========
    parser.add_argument(
        "--fedprox_mu",
        type=float,
        default=0.01,
        help="FedProx proximal regularization coefficient μ"
    )

    # ========= Ditto options =========
    parser.add_argument(
        "--ditto_lambda",
        type=float,
        default=1.0,
        help="Ditto personalization regularization coefficient λ"
    )
    parser.add_argument(
        "--ditto_personal_epochs",
        type=int,
        default=None,
        help="Ditto personalized-model training epochs (defaults to local_epochs)."
    )

    # ========= FedRep options =========
    parser.add_argument(
        "--fedrep_head_epochs",
        type=int,
        default=1,
        help="FedRep classifier-head training epochs."
    )
    parser.add_argument(
        "--fedrep_rep_epochs",
        type=int,
        default=4,
        help="FedRep representation-layer training epochs."
    )

    # ========= FedPAC options =========
    parser.add_argument(
        "--fedpac_lam",
        type=float,
        default=1.0,
        help="FedPAC prototype-alignment loss weight."
    )
    parser.add_argument(
        "--fedpac_head_epochs",
        type=int,
        default=1,
        help="FedPAC classifier-head training epochs."
    )
    parser.add_argument(
        "--fedpac_feature_epochs",
        type=int,
        default=4,
        help="FedPAC feature-extractor training epochs."
    )

    # ========= FedALA options =========
    parser.add_argument(
        "--fedala_eta",
        type=float,
        default=0.1,
        help="FedALA aggregation-weight learning rate."
    )
    parser.add_argument(
        "--fedala_layer_idx",
        type=int,
        default=-1,
        help="FedALA starting aggregation layer index (-1 selects the final layer)."
    )
    parser.add_argument(
        "--fedala_num_pre_loss",
        type=int,
        default=5,
        help="Number of samples used by FedALA to compute aggregation weights."
    )
    parser.add_argument(
        "--fedala_adaptive_lr",
        type=float,
        default=0.01,
        help="FedALA adaptive learning rate."
    )
    parser.add_argument(
        "--fedala_rand_percent",
        type=int,
        default=80,
        help="FedALA data-sampling ratio."
    )
    parser.add_argument(
        "--fedala_threshold",
        type=float,
        default=0.1,
        help="FedALA convergence threshold."
    )

    # ========= FedAPA options =========
    parser.add_argument(
        "--fedapa_server_lr",
        type=float,
        default=0.1,
        help="FedAPA server learning rate."
    )
    parser.add_argument(
        "--fedapa_clamp_min",
        type=float,
        default=0.5,
        help="FedAPA minimum weight."
    )
    parser.add_argument(
        "--fedapa_clamp_max",
        type=float,
        default=1.0,
        help="FedAPA maximum weight."
    )

    # ========= FedSelect options =========
    parser.add_argument(
        "--fedselect_keep_ratio",
        type=float,
        default=0.8,
        help="FedSelect: fraction of currently shared parameters to retain as globally shared (mask=0) after each update.",
    )
    parser.add_argument(
        "--fedselect_update_interval",
        type=int,
        default=3,
        help="FedSelect: mask-update interval in federated rounds.",
    )

    # ========= FedAS options =========
    parser.add_argument(
        "--fedas_align_weight",
        type=float,
        default=0.5,
        help="FedAS: parameter-alignment weight (0.0-1.0)."
    )
    parser.add_argument(
        "--fedas_sync_threshold",
        type=float,
        default=0.1,
        help="FedAS: client synchronization threshold."
    )
    parser.add_argument(
        "--wo_local",
        action="store_true",
        default=False,
        help="FedAS: disable local prototype alignment and use global-model parameters directly."
    )
    parser.add_argument(
        "--wo_global",
        action="store_true",
        default=False,
        help="FedAS: disable FIM-weighted aggregation and use standard FedAvg aggregation."
    )
    parser.add_argument(
        "--yxy_hyper",
        type=float,
        default=1.0,
        help="FedAS hyperparameter."
    )

    # ========= FedLTN options =========
    parser.add_argument(
        "--lth_norm_layer_keywords",
        type=str,
        nargs="+",
        default=["norm", "bn", "ln", "gn"],
        help="FedLTN: keyword identifying normalization layers whose parameters are excluded from aggregation."
    )
    parser.add_argument(
        "--lth_server_momentum",
        type=float,
        default=0.0,
        help="FedLTN: server momentum coefficient for faster aggregation convergence."
    )
    parser.add_argument(
        "--lth_acc_threshold",
        type=float,
        default=0.5,
        help="FedLTN: validation-accuracy threshold for triggering pruning."
    )
    parser.add_argument(
        "--lth_prune_step",
        type=float,
        default=0.1,
        help="FedLTN: increase in the local-parameter ratio per pruning step."
    )
    parser.add_argument(
        "--lth_target_sparsity",
        type=float,
        default=0.5,
        help="FedLTN: target local-parameter ratio (fraction with mask=1)."
    )
    parser.add_argument(
        "--lth_use_jumpstart",
        action="store_true",
        default=False,
        help="FedLTN: whether to enable Jump-Start warm-up."
    )
    parser.add_argument(
        "--lth_jumpstart_epochs",
        type=int,
        default=5,
        help="FedLTN: Local-training epochs during Jump-Start warm-up."
    )

    # ========= Local method options =========
    parser.add_argument(
        "--local_patience",
        type=int,
        default=5,
        help="Local-method early-stopping patience based on validation AUC."
    )

    # ========= Aggregation options =========
    parser.add_argument(
        "--weighting_method",
        type=str,
        default="fedavg",
        choices=["fedavg", "entropy"],
        help="Client aggregation-weighting strategy: fedavg / entropy.",
    )
    parser.add_argument(
        "--aggregation_scope",
        type=str,
        default="all",
        choices=["masked_only", "all"],
        help="Aggregation scope: masked_only aggregates only mask=0 parameters; all aggregates all parameters.",
    )
    parser.add_argument(
        "--entropy_update_view",
        type=str,
        choices=["scope_aware", "shared_mask0"],
        default="scope_aware",
        help=(
            "Update coordinates used by entropy alignment/stability. "
            "scope_aware preserves legacy behavior; shared_mask0 decouples "
            "client reliability weights from the HA aggregation decision."
        ),
    )

    parser.add_argument(
        "--enable_weight_protection",
        action="store_true",
        help="Whether to enable client aggregation-weight protection (disabled by default)."
    )

    parser.add_argument(
        "--agg_weight_min",
        type=float,
        default=0.03,
        help="Minimum client aggregation weight."
    )

    parser.add_argument(
        "--agg_weight_smooth",
        type=float,
        default=0.1,
        help="Smoothing coefficient toward uniform client weights."
    )

    # ========= Personalization settings =========
    parser.add_argument("--max_metrics_history", type=int, default=10)
    parser.add_argument(
        "--kdi_alpha",
        type=float,
        default=0.7,
        help="KDARS KDI weight for normalized knowledge drift.",
    )
    parser.add_argument(
        "--personal_min_ratio",
        type=float,
        default=0.2,
        help=(
            "KDARS minimum per-update personalization rate and lower bound "
            "for the cumulative local-parameter ratio."
        ),
    )
    parser.add_argument(
        "--personal_max_ratio",
        type=float,
        default=0.8,
        help=(
            "KDARS maximum per-update personalization rate and strict upper "
            "bound for the cumulative local-parameter ratio."
        ),
    )

    # ========= Parameter-level personalization / pruning =========
    parser.add_argument(
        "--pruning_method",
        type=str,
        default="none",
        choices=["none", "kdars"],
        help=(
            "Parameter-level personalization method. "
            "kdars uses Knowledge Drift-Aware Adaptive Personalization (KDARS)."
        ),
    )
    parser.add_argument(
        "--lth_epoch_iters",
        type=int,
        default=2,
        help="Parameter-level personalization / mask-update interval in federated rounds.",
    )
    parser.add_argument("--epsilon", type=float, default=1e-8)

    # ========= Auxiliary one-time aggregation-scope selector =========
    parser.add_argument(
        "--enable_ha_scope",
        action="store_true",
        default=False,
        help="Enable the auxiliary one-time aggregation-scope selector (FedDyna-HA). Disabled by default.",
    )
    parser.add_argument(
        "--ha_select_threshold",
        type=float,
        default=0.5,
        help="At the first scheduled mask-update round, the selector chooses masked_only when the update-disagreement score reaches the threshold; otherwise it chooses all. The selected scope remains fixed.",
    )
    parser.add_argument(
        "--ha_update_basis",
        type=str,
        choices=["train_delta"],
        default="train_delta",
        help="Selector client-update reference: train_delta is the difference between the locally trained and global models.",
    )
    parser.add_argument(
        "--ha_scope_mode",
        type=str,
        choices=["robust"],
        default="robust",
        help="Robust update-direction summary: mean - λ_std·std - λ_q·(mean-q) - λ_neg·negative_ratio.",
    )
    parser.add_argument(
        "--ha_quantile",
        type=float,
        default=0.1,
        help="HA low quantile used to compute low_quantile_cos.",
    )
    parser.add_argument(
        "--ha_lambda_std",
        type=float,
        default=0.3,
        help="Standard-deviation penalty coefficient for robust HA mode.",
    )
    parser.add_argument(
        "--ha_lambda_q",
        type=float,
        default=0.3,
        help="(Mean - quantile) penalty coefficient for robust HA mode.",
    )
    parser.add_argument(
        "--ha_lambda_neg",
        type=float,
        default=0.4,
        help="Negative-ratio penalty coefficient for robust HA mode.",
    )
    parser.add_argument(
        "--ha_save_pairwise",
        action="store_true",
        default=False,
        help="Whether HA saves pairwise cosine details for the selection round to CSV.",
    )
    parser.add_argument(
        "--ha_debug",
        action="store_true",
        default=False,
        help="Whether HA prints detailed diagnostic information.",
    )

    parser.add_argument(
        "--save_ha_analysis",
        action="store_true",
        default=False,
        help="Save passive per-round HA analysis artifacts without changing training.",
    )
    parser.add_argument(
        "--ha_analysis_dir",
        type=str,
        default=None,
        help="Output directory for HA analysis artifacts. Defaults to save_dir/exp_name/ha_analysis.",
    )

    # ========= Debugging / visualization =========
    parser.add_argument("--verbose", type=int, default=1, help="0=quiet, 1=normal, 2=verbose")

    parser.add_argument("--log_batch_interval", type=int, default=10)
    parser.add_argument("--save_plots", action="store_true", help="Save client training-curve plots.")
    parser.add_argument("--plot_dir", type=str, default="./save/plots")
    parser.add_argument("--print_param_stats", action="store_true", help="Print parameter norms and changes.")
    parser.add_argument("--debug_client_limit", type=int, default=-1, help="Debug only the first N clients; -1 means all clients.")

    return parser


def parse_args() -> argparse.Namespace:
    args = build_parser().parse_args()

    # Normalize basic text fields to avoid downstream branching differences caused by capitalization.
    if isinstance(args.dataset, str):
        args.dataset = args.dataset.strip().lower()
    if isinstance(args.model, str):
        args.model = args.model.strip().lower()
    if isinstance(args.device, str):
        args.device = args.device.strip().lower()

    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.local_bs <= 0:
        raise ValueError("local_bs must be > 0")
    if args.local_epochs <= 0:
        raise ValueError("local_epochs must be > 0")
    if args.com_rounds <= 0:
        raise ValueError("com_rounds must be > 0")
    if args.lr <= 0:
        raise ValueError("lr must be > 0")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be >= 0")
    if args.dropout < 0.0 or args.dropout >= 1.0:
        raise ValueError("dropout must satisfy 0 <= dropout < 1")
    if args.clipgradnorm <= 0:
        raise ValueError("clipgradnorm must be > 0")
    if args.max_metrics_history <= 0:
        raise ValueError("max_metrics_history must be > 0")
    if not (0.0 <= args.fedselect_keep_ratio <= 1.0):
        raise ValueError("fedselect_keep_ratio must satisfy 0 <= fedselect_keep_ratio <= 1")
    if args.fedselect_update_interval <= 0:
        raise ValueError("fedselect_update_interval must be > 0")
    if not (0.0 <= args.kdi_alpha <= 1.0):
        raise ValueError("kdi_alpha must satisfy 0 <= kdi_alpha <= 1")
    if not (0.0 <= args.personal_min_ratio <= 1.0):
        raise ValueError("personal_min_ratio must satisfy 0 <= personal_min_ratio <= 1")
    if not (0.0 <= args.personal_max_ratio <= 1.0):
        raise ValueError("personal_max_ratio must satisfy 0 <= personal_max_ratio <= 1")
    if args.personal_min_ratio > args.personal_max_ratio:
        raise ValueError("personal_min_ratio cannot be greater than personal_max_ratio")
    if args.lth_epoch_iters <= 0:
        raise ValueError("lth_epoch_iters must be > 0")
    if args.epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    # Validate HA parameters
    if getattr(args, "enable_ha_scope", False):
        # Selector parameters
        if not (0.0 <= args.ha_select_threshold <= 1.0):
            raise ValueError("ha_select_threshold must satisfy 0 <= threshold <= 1")
        # Update-direction summary parameters
        if not (0.0 < args.ha_quantile < 1.0):
            raise ValueError("ha_quantile must satisfy 0 < ha_quantile < 1")
        if args.ha_lambda_std < 0 or args.ha_lambda_q < 0 or args.ha_lambda_neg < 0:
            raise ValueError("ha_lambda_std / ha_lambda_q / ha_lambda_neg must be >= 0")


def args_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    return vars(args).copy()


def dump_effective_args(args: argparse.Namespace, save_dir: str, filename: str = "effective_config.json") -> str:
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(args_to_dict(args), f, ensure_ascii=False, indent=2)
    return save_path

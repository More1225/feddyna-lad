import copy

from federated.lottery_ticket import (
    init_mask_zeros,
    ratio_based_update,
)
from utils.debug_tools import log


class MaskManager:
    def __init__(self, model, args):
        self.args = args
        self.init_mask = init_mask_zeros(model)

    def init_client_masks(self, client_ids):
        return {cid: copy.deepcopy(self.init_mask) for cid in client_ids}

    def should_update(self, round_num: int) -> bool:
        from utils.debug_tools import log
        result = (
            self.args.pruning_method != "none"
            and round_num != 0
            and round_num % self.args.lth_epoch_iters == 0
        )
        log(f"[MaskManager] should_update round={round_num}, pruning_method={self.args.pruning_method}, lth_epoch_iters={self.args.lth_epoch_iters}, result={result}", level=2)
        return result

    def update_client_mask(
        self,
        current_model,
        prev_model,
        current_mask,
        local_metrics=None,
        global_metrics=None,
        global_ratio_override=None,
    ):
        current_state = current_model.state_dict()
        prev_state = prev_model.state_dict()

        log(f"[MaskManager] Updating mask with method: {self.args.pruning_method}", 2, self.args)

        if self.args.pruning_method == "kdars":
            if global_ratio_override is None:
                return current_mask
            return ratio_based_update(
                current_state_dict=current_state,
                prev_state_dict=prev_state,
                current_mask=current_mask,
                global_ratio=global_ratio_override,
                args=self.args,
            )

        return current_mask

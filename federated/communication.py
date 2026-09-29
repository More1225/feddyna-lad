"""Estimated communication payload for the existing single-process protocol.

No serialization or network measurement. Full state dictionaries cross the
logical client/server boundary; aggregation masks never reduce model payload.
Server owns aggregation, mask updates and FedAPA personalized aggregation.
"""
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch

CATEGORIES = ("model", "mask", "indices", "auxiliary", "prototype")
BASIS = "estimated communication payload"
VERSION = "dense_protocol_v2"
ASSUMPTIONS = [
    "Single-process simulation; logical client/server boundaries, not network measurements.",
    "Full state_dict upload and global download, including heads/buffers; no sparse encoding.",
    "Tensor bytes use actual dtype; masks retain their current tensor dtype, no bit packing.",
    "Server owns aggregation and mask updates; update-round masks are sent densely to clients.",
    "All-zero initial masks are reconstructed locally from the shared schema, not transmitted.",
    "Python int/float scalars and prototype class IDs assumed int64/float64; bool is one byte.",
    "Schema field names/model keys and transport addressing are pre-agreed; framing/serialization/network overhead excluded.",
    "Sample counts sent each round; entropy/simple-F1 upload delta_f1; alignment/HA/KDARS computed server-side.",
    "Global post-aggregation model is reused for personal training and round-end initialization, counted once.",
    "Final round global download is included. Prototype downloads counted on actual pre-training use only.",
    "FedAPA server-side personalized state adds one model download in rounds after cache initialization.",
]

def payload_bytes(obj, seen=None):
    """Logical numeric payload; exact repeated tensor views/containers count once/message."""
    seen = set() if seen is None else seen
    if obj is None:
        return 0
    if torch.is_tensor(obj):
        if obj.layout != torch.strided:
            raise TypeError("Sparse encoding requires an explicit audited message schema")
        key = ("tensor", str(obj.device), obj.untyped_storage().data_ptr(),
               obj.storage_offset(), tuple(obj.shape), tuple(obj.stride()), obj.dtype)
        if key in seen:
            return 0
        seen.add(key)
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, np.ndarray):
        if obj.dtype.hasobject:
            raise TypeError("Object arrays need an explicit numeric schema")
        key = ("numpy", id(obj))
        if key in seen:
            return 0
        seen.add(key)
        return int(obj.nbytes)
    if isinstance(obj, (dict, list, tuple)):
        key = ("container", id(obj))
        if key in seen:
            return 0
        seen.add(key)
        values = obj.values() if isinstance(obj, dict) else obj
        return sum(payload_bytes(value, seen) for value in values)
    if isinstance(obj, np.generic):
        return int(obj.dtype.itemsize)
    if isinstance(obj, bool):
        return 1
    if isinstance(obj, int):
        if not -(2**63) <= obj < 2**63:
            raise OverflowError("Assumed int64 scalar out of range")
        return 8
    if isinstance(obj, float):
        return 8
    raise TypeError(f"Unspecified payload encoding: {type(obj).__name__}")

class CommunicationTracker:
    def __init__(self, args=None, num_clients=None):
        self.args = args
        self.num_clients = num_clients
        self.events = []
        self.round_communication = []
        self.training_start_time = None
        self.training_end_time = None
        self.total_training_time = 0.0
        self.initialized = False
        self._ids = set()
        self.warnings = []

    def _message(self, round_num, cid, direction, name, **components):
        identity = (round_num, cid, direction, name)
        if identity in self._ids:
            raise ValueError(f"Duplicate message: {identity}")
        if direction not in ("upload", "download"):
            raise ValueError(direction)
        self._ids.add(identity)
        seen = set()
        counts = {key: payload_bytes(components.get(key), seen) for key in CATEGORIES}
        self.events.append(dict(round=round_num, client_id=int(cid), direction=direction,
                                message=name, **{k+"_bytes":v for k,v in counts.items()},
                                total_bytes=sum(counts.values())))

    def record_initialization(self, global_state, client_ids):
        if self.initialized:
            raise ValueError("Initialization already counted")
        if self.num_clients is None:
            self.num_clients = len(client_ids)
        for cid in client_ids:
            self._message(-1, cid, "download", "initial_model", model=global_state)
        self.initialized = True

    def record_pre_round(self, round_num, client_ids, fl_method=None):
        if not self.initialized:
            raise ValueError("Record initialization first")
        name = getattr(self.args, "fl_method", "fedavg")
        if name != "fedavg" and fl_method is None:
            warning = "Requested adapter unavailable; runner used fallback. Not valid for method comparison."
            if warning not in self.warnings:
                self.warnings.append(warning)
        if name == "fedpac" and fl_method is not None:
            protos = fl_method.global_protos
            if protos:
                for cid in client_ids:
                    self._message(round_num, cid, "download", "global_prototypes",
                                  prototype=list(protos.values()),
                                  indices=np.asarray(list(protos), dtype=np.int64))
    def record_personalized_state(self, round_num, cid, personalized_state):
        """Called where FedAPA consumes the actual server-produced state."""
        self._message(round_num, cid, "download", "personalized_server_model",
                      model=personalized_state)

    def record_round(self, round_num, client_state_dicts, global_state, client_masks,
                     client_weights=None, fl_method_name="fedavg", classifier_keys=None,
                     prototype_bytes=0, client_results=None, fl_method=None,
                     mask_updated=False):
        if not self.initialized:
            raise ValueError("Record initialization first")
        if any(r["round"] == round_num for r in self.round_communication):
            raise ValueError("Round already recorded")
        if prototype_bytes:
            raise ValueError("Use actual per-client prototypes, not a guessed aggregate")
        client_results = client_results or {}
        weighting = getattr(self.args, "weighting_method", "fedavg")
        for cid, state in client_state_dicts.items():
            aux = {}
            result = client_results.get(cid)
            if result is not None:
                stats = result["train_result"]["data_stats"]
                if weighting in ("fedavg", "entropy"):
                    aux["sample_count"] = stats["sample_count"]
                if weighting == "entropy":
                    aux["delta_f1"] = float(result["delta_f1"])
            if fl_method_name == "fedas" and cid in getattr(fl_method, "client_fim", {}):
                aux["fim_trace"] = fl_method.client_fim[cid]
            self._message(round_num, cid, "upload", "client_model_and_metrics",
                          model=state, auxiliary=aux)
            self._message(round_num, cid, "download", "post_aggregation_global", model=global_state)
            if mask_updated:
                self._message(round_num, cid, "download", "server_mask_update", mask=client_masks[cid])
            # FedLTN does not currently produce or consume prototypes. Only count
            # a local prototype payload if that hook actually populated it.
            protos = (getattr(fl_method, "local_protos", {}) if fl_method_name == "fedpac"
                      else getattr(fl_method, "client_prototypes", {}) if fl_method_name == "fedltn" else {})
            if cid in protos and protos[cid]:
                local = protos[cid]
                self._message(round_num, cid, "upload", "local_prototypes",
                              prototype=list(local.values()),
                              indices=np.asarray(list(local), dtype=np.int64))
        row = self._totals(e for e in self.events if e["round"] == round_num)
        n = len(client_state_dicts)
        row.update(round=round_num, fl_method=fl_method_name, num_clients=n,
                   participation_rate=n/self.num_clients if self.num_clients else 0,
                   accounting_basis=BASIS, accounting_version=VERSION, mb_divisor=10**6,
                   uplink_mb=row["upload_bytes"]/1e6, downlink_mb=row["download_bytes"]/1e6,
                   total_mb=row["total_bytes"]/1e6,
                   avg_uplink_per_client_mb=row["upload_bytes"]/1e6/n if n else 0,
                   avg_downlink_per_client_mb=row["download_bytes"]/1e6/n if n else 0)
        self.round_communication.append(row)
        return row

    @staticmethod
    def _totals(events):
        totals = {"upload_bytes":0, "download_bytes":0, "total_bytes":0}
        for direction in ("upload", "download"):
            for category in CATEGORIES:
                totals[direction+"_"+category+"_bytes"] = 0
        for event in events:
            direction = event["direction"]
            totals[direction+"_bytes"] += event["total_bytes"]
            for category in CATEGORIES:
                totals[direction+"_"+category+"_bytes"] += event[category+"_bytes"]
        totals["total_bytes"] = totals["upload_bytes"] + totals["download_bytes"]
        return totals

    def get_summary(self):
        cumulative = self._totals(self.events)
        initial = self._totals(e for e in self.events if e["round"] == -1)
        rounds = self._totals(e for e in self.events if e["round"] >= 0)
        n = len(self.round_communication)
        args = self.args
        return dict(
            **cumulative, initialization=initial, rounds_only=rounds,
            accounting_basis=BASIS, accounting_version=VERSION, mb_divisor=10**6,
            method=getattr(args,"method_tag","") or getattr(args,"fl_method","fedavg"),
            fl_method=getattr(args,"fl_method","fedavg"),
            scenario=getattr(args,"scenario_tag",""), seed=getattr(args,"seed",None),
            num_clients=self.num_clients, participation_rate=(
                self.round_communication[0]["participation_rate"] if n and
                len({r["participation_rate"] for r in self.round_communication})==1 else None),
            num_rounds=n, completed_rounds=n, requested_rounds=getattr(args,"com_rounds",None),
            initialization_policy="one full model per client before round 0; included separately",
            final_download_included=True, assumptions=ASSUMPTIONS,
            protocol_valid=not self.warnings, warnings=list(self.warnings),
            total_uplink_mb=cumulative["upload_bytes"]/1e6,
            total_downlink_mb=cumulative["download_bytes"]/1e6,
            total_communication_mb=cumulative["total_bytes"]/1e6,
            avg_uplink_per_round_mb=rounds["upload_bytes"]/1e6/n if n else 0,
            avg_downlink_per_round_mb=rounds["download_bytes"]/1e6/n if n else 0,
            total_training_time_sec=self.total_training_time,
            total_training_time_min=self.total_training_time/60)

    def export(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        summary = self.get_summary()
        (directory/"communication_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (directory/"communication_config.json").write_text(
            json.dumps(vars(self.args) if self.args else {}, ensure_ascii=False, indent=2), encoding="utf-8")
        for name, rows in (("communication_rounds.csv",self.round_communication),
                           ("communication_events.csv",self.events)):
            if rows:
                with (directory/name).open("w",newline="",encoding="utf-8") as handle:
                    writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)

    def compute_state_dict_size(self, state_dict):
        return payload_bytes(state_dict)

    def start_training(self):
        self.training_start_time = time.time()

    def end_training(self):
        self.training_end_time = time.time()
        if self.training_start_time is not None:
            self.total_training_time = self.training_end_time-self.training_start_time
        return self.total_training_time

    def reset(self):
        self.__init__(self.args,self.num_clients)

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Any


class ClientMetricsTracker:
    def __init__(self, client_ids, max_history: int = 10):
        self.client_ids = list(client_ids)
        self.max_history = max_history

        self.f1_history = {cid: deque(maxlen=max_history) for cid in self.client_ids}
        self.latest_metrics: Dict[int, Dict[str, float]] = {}
        self.data_stats: Dict[int, Dict[str, Any]] = defaultdict(dict)

    def update_round(self, client_id: int, metrics: Dict[str, float], data_stats: Dict[str, Any] | None = None):
        self.latest_metrics[client_id] = metrics
        self.f1_history[client_id].append(float(metrics.get("f1", 0.0)))
        if data_stats is not None:
            self.data_stats[client_id] = data_stats

    def get_client_f1_history(self, client_id: int):
        return list(self.f1_history.get(client_id, []))

    def get_latest_metrics(self, client_id: int):
        return self.latest_metrics.get(client_id, {})

    def get_client_data_stats(self, client_id: int):
        return self.data_stats.get(client_id, {})

    def get_all_client_ids(self):
        return self.client_ids

    def get_round_snapshot(self, client_id: int):
        return {
            "latest_metrics": self.get_latest_metrics(client_id),
            "data_stats": self.get_client_data_stats(client_id),
            "f1_history": self.get_client_f1_history(client_id),
        }

    def reset(self):
        self.f1_history = {cid: deque(maxlen=self.max_history) for cid in self.client_ids}
        self.latest_metrics.clear()
        self.data_stats.clear()


class GlobalMetricsTracker:
    def __init__(self):
        self.history = []

    def update_round(self, metrics: Dict[str, float]):
        self.history.append(metrics)

    def get_latest_metrics(self):
        return self.history[-1] if self.history else {}

    def reset(self):
        self.history.clear()
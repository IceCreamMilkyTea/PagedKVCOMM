from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Any, Dict
import os


@dataclass(frozen=True)
class KVCommConfig:
    """
    Configuration for KV communication and scheduling.

    Args:
        threshold (float): The threshold for the KV communication.
        thread_pool_workers (int): The number of threads to use for the thread pool.
        worker_timeout (float): The timeout for the worker.
    """
    threshold: float = 0.3
    max_anchor_num: int = 20
    window_size: int = 5
    thread_pool_workers: int = 8
    worker_timeout: float = 30.0
    use_local_reference: bool = False
    local_ref_mode: str = "no_check"  # "no_check", "cross_delta_consistency", "weight_confidence"
    local_ref_consistency_threshold: float = 0.5  # max relative std for cross-delta consistency
    local_ref_weight_threshold: float = 0.3  # min max-weight for weight confidence
    proactive_evict_threshold: float = 0.15  # evict anchors when free pool fraction falls below this
    use_current_round_sharing: bool = True   # use upstream agent's current-round delta for user_question
    crs_priority: bool = False               # bypass dense_prefill for user_question anchors even without own delta (ablation)

    @classmethod
    def from_env(cls) -> "KVCommConfig":
        """Create a config from environment variables with safe defaults."""
        return cls(
            threshold=float(os.environ.get("THRESHOLD", cls.threshold)),
            max_anchor_num=int(os.environ.get("MAX_ANCHOR_NUM", cls.max_anchor_num)),
            window_size=int(os.environ.get("WINDOW_SIZE", cls.window_size)),
            thread_pool_workers=int(os.environ.get("KVCOMM_THREAD_WORKERS", cls.thread_pool_workers)),
            worker_timeout=float(os.environ.get("KVCOMM_WORKER_TIMEOUT", cls.worker_timeout)),
            use_local_reference=os.environ.get("KVCOMM_LOCAL_REF", "0") == "1",
            local_ref_mode=os.environ.get("KVCOMM_LOCAL_REF_MODE", cls.local_ref_mode),
            local_ref_consistency_threshold=float(os.environ.get("KVCOMM_LOCAL_REF_CONSISTENCY", cls.local_ref_consistency_threshold)),
            local_ref_weight_threshold=float(os.environ.get("KVCOMM_LOCAL_REF_WEIGHT", cls.local_ref_weight_threshold)),
            proactive_evict_threshold=float(os.environ.get("KVCOMM_PROACTIVE_EVICT", cls.proactive_evict_threshold)),
            use_current_round_sharing=os.environ.get("KVCOMM_CURRENT_ROUND_SHARING", "1") == "1",
            crs_priority=os.environ.get("KVCOMM_CRS_PRIORITY", "0") == "1",
        ).validate()

    def apply_overrides(self, **overrides: Any) -> "KVCommConfig":
        """Return a copy with provided non-None fields overridden."""
        current: Dict[str, Any] = asdict(self)
        for key, value in overrides.items():
            if value is None or key not in current:
                continue
            current[key] = value
        return replace(self, **current).validate()

    def validate(self) -> "KVCommConfig":
        """Validate value ranges and return self."""
        if self.thread_pool_workers <= 0:
            raise ValueError("thread_pool_workers must be positive")
        if self.worker_timeout <= 0:
            raise ValueError("worker_timeout must be positive")
        return self

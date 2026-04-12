import os
from dataclasses import dataclass
from typing import Literal
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    paged_backend: Literal["paged", "radix"] = "paged"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.paged_backend in ("paged", "radix")
        try:
            self.hf_config = AutoConfig.from_pretrained(self.model)
        except ValueError:
            # Older transformers may not support newer rope_scaling schemas.
            import json as _json

            cfg_path = os.path.join(self.model, "config.json")
            with open(cfg_path, "r", encoding="utf-8") as handle:
                raw = _json.load(handle)
            raw.pop("rope_scaling", None)
            from transformers import LlamaConfig

            self.hf_config = LlamaConfig(**raw)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len

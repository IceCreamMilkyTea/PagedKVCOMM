import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()

        # Greedy path for rows with near-zero temperature.
        greedy_mask = temperatures <= 1e-10
        greedy_tokens = logits.argmax(dim=-1)

        # Sampling path for rows with positive temperature.
        safe_temperatures = temperatures.clamp_min(1e-10)
        scaled_logits = logits.div(safe_temperatures.unsqueeze(dim=1))
        probs = torch.softmax(scaled_logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)

        return torch.where(greedy_mask, greedy_tokens, sample_tokens)

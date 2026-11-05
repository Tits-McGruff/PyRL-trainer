"""Neural network policy and value heads."""

from typing import List, Tuple
import math

import torch
from torch import nn

POLICY_PARAMETERIZATION_VERSION = 2
MIN_TURN_LOG_STD = -5.0
MAX_TURN_LOG_STD = 1.0


class PolicyValueNet(nn.Module):
    """
    Simple MLP policy/value head.
    Outputs:
      - turn_mean for the pre-tanh Normal policy
      - boost_logit
      - value scalar
    """
    def __init__(
        self, obs_dim: int, hidden: int = 256, layers: int = 2,
        initial_turn_std: float = 0.50,
    ):
        super().__init__()
        if (isinstance(initial_turn_std, bool) or not math.isfinite(initial_turn_std)
                or not math.exp(MIN_TURN_LOG_STD)
                <= initial_turn_std <= math.exp(MAX_TURN_LOG_STD)):
            raise ValueError("initial steering standard deviation exceeds numerical bounds")
        self.turn_log_std = nn.Parameter(torch.tensor(math.log(initial_turn_std)))
        self.constrain_turn_std()
        layers = int(max(1, layers))

        blocks: List[nn.Module] = []
        blocks.append(nn.Linear(obs_dim, hidden))
        blocks.append(nn.ReLU())

        for _ in range(layers - 1):
            blocks.append(nn.Linear(hidden, hidden))
            blocks.append(nn.ReLU())

        self.net = nn.Sequential(*blocks)
        self.turn_head = nn.Linear(hidden, 1)
        self.boost_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def steering_std(self) -> torch.Tensor:
        """Trainable state-independent latent Gaussian scale with numerical guards."""
        return self.turn_log_std.clamp(MIN_TURN_LOG_STD, MAX_TURN_LOG_STD).exp()

    @torch.no_grad()
    def constrain_turn_std(self) -> None:
        """Project strictly inside the clamp; some backends give zero edge gradients."""
        lower = self.turn_log_std.new_tensor(MIN_TURN_LOG_STD)
        upper = self.turn_log_std.new_tensor(MAX_TURN_LOG_STD)
        self.turn_log_std.clamp_(torch.nextafter(lower, upper), torch.nextafter(upper, lower))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute pre-tanh turn mean, boost logit, and value."""
        x = self.net(obs)
        turn_mean = self.turn_head(x).squeeze(-1)
        boost_logit = self.boost_head(x).squeeze(-1)
        value = self.value_head(x).squeeze(-1)
        return turn_mean, boost_logit, value

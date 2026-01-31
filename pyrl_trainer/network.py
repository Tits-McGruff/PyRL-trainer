import torch
from torch import nn
from typing import Tuple, List


class PolicyValueNet(nn.Module):
    """
    Simple MLP policy/value head.
    Outputs:
      - turn_mean in [-1, 1] via tanh
      - boost_logit, converted to probability with sigmoid
      - value scalar
    """
    def __init__(self, obs_dim: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        layers = int(max(1, layers))

        blocks: List[nn.Module] = []
        # First layer maps obs -> hidden
        blocks.append(nn.Linear(obs_dim, hidden))
        blocks.append(nn.ReLU())

        # Additional hidden layers (hidden -> hidden)
        for _ in range(layers - 1):
            blocks.append(nn.Linear(hidden, hidden))
            blocks.append(nn.ReLU())

        self.net = nn.Sequential(*blocks)
        self.turn_head = nn.Linear(hidden, 1)
        self.boost_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        turn_mean = self.turn_head(x).squeeze(-1)
        boost_logit = self.boost_head(x).squeeze(-1)
        value = self.value_head(x).squeeze(-1)
        return turn_mean, boost_logit, value

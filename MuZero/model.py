import typing
from typing import Dict, List

from *** import Action

import torch.nn as nn
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]


class Network(ABC, nn.module):
    def __init__(self, scaling=False, down_sample=False):
        super().__init__()
        self.steps = 0
        self.scaling = scaling
        self.down_sample = down_sample

    @abstractmethod
    def representation(self, image):
        pass

    @abstractmethod
    def dynamics(self, hidden_state, action):
        pass

    @abstractmethod
    def prediction(self, hidden_state):
        pass

    def initial_inference(self, image) -> NetworkOutput:
        hidden_state = self.representation(image)
        policy_logits, value = self.prediction(hidden_state)

        return NetworkOutput(value, 0, policy_logits, hidden_state)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        next_state, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_state)

        return NetworkOutput(value, reward, policy_logits, next_state)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def training_steps(self) -> int:
        return self.steps

    # def scaling_transform(self, x, support_size, epsilon=1e-3):
    #
    #     x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1 + epsilon * x)
    #
    #     x = torch.clamp(x, -support_size, support_size)
    #     p_low = x - x.floor()
    #     p_high = 1 - p_low
    #
    #     logits_target = torch.zeros(x.shape[0], x.shape[1], 2*support_size+1)
    #     ...


# MuZero Network Structure with Fully Connected layers
class FullyConnectedNetwork(Network):
    def __init__(self):
        super().__init__()
        self.prediction_policy_network =
        self.prediction_value_network =
        self.representation_network =
        self.dynamics_state_network =
        self.dynamics_reward_network =

    def prediction(self, hidden_state):
        policy_logits = self.prediction_policy_network(hidden_state)
        value = self.prediction_value_network(hidden_state)
        return policy_logits, value

    def representation(self, image):

    def dynamics(self, hidden_state, action):


class ResBlock(nn.Module):
    def __init__(self, num_channels=256, stride=1, shortcut=None):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(True),
            nn.Conv2d(num_channels, num_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(num_channels),
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


# The Down Sample part for Atari Observation to reduce spatial resolution.
# Input size BatchSize * 128 * 96 * 96
# Output size BatchSize * 256 * 6 * 6
class DownSampleHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2),
            ResBlock(num_channels=128),
            ResBlock(num_channels=128),
            nn.Conv2d(128, 256, 3, stride=2),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            nn.AvgPool2d(2),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.net(x)


# MuZero Network Structure with Residual Blocks
class ResidualNetwork(Network):
    def __init__(self, num_channels, num_blocks):
        super().__init__()
        self.prediction_policy_network =
        self.prediction_value_network =
        self.representation_network =
        self.dynamics_state_network =
        self.dynamics_reward_network =

    def prediction(self, hidden_state):
        policy_logits = self.prediction_policy_network(hidden_state)
        value = self.prediction_value_network(hidden_state)
        return policy_logits, value

    def representation(self, image):

    def dynamics(self, hidden_state, action):


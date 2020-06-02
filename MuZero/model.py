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
    def __init__(self):
        super().__init__()
        self.steps = 0

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


class ResidualNetwork(Network):
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
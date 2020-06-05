import typing
from typing import Dict, List
from game_env import Action

import torch.nn as nn
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]

class UniformNetwork():
    """policy -> uniform, value -> 0, reward -> 0"""

    def __init__(self, action_size):
        self.action_size = action_size

    def initial_inference(self, image):
        return NetworkOutput(0, 0, {Action(i): 1 / self.action_size for i in range(self.action_size)}, None)

    def recurrent_inference(self, hidden_state, action):
        return NetworkOutput(0, 0, {Action(i): 1 / self.action_size for i in range(self.action_size)}, None)

    def training_steps(self):
        pass

class Network(ABC, nn.Module):
    def __init__(self, scaling=False, down_sample=False):
        super().__init__()
        self.steps = 0
        # whether to do scalar transform or down sample
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

    def initial_inference(self, image):
        hidden_state = self.representation(image)
        policy_logits, value = self.prediction(hidden_state)

        return NetworkOutput(value, 0, policy_logits, hidden_state)

    def recurrent_inference(self, hidden_state, action):
        next_state, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_state)

        return NetworkOutput(value, reward, policy_logits, next_state)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    """
    def training_steps(self) -> int:
        return self.steps
    """
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
"""
class BaseMuZeroNet(nn.Module):
    def __init__(self, inverse_value_transform, inverse_reward_transform):
        super(BaseMuZeroNet, self).__init__()
        self.inverse_value_transform = inverse_value_transform
        self.inverse_reward_transform = inverse_reward_transform

    def prediction(self, state):
        raise NotImplementedError

    def representation(self, obs_history):
        raise NotImplementedError

    def dynamics(self, state, action):
        raise NotImplementedError

    def initial_inference(self, obs) -> NetworkOutput:
        state = self.representation(obs)
        actor_logit, value = self.prediction(state)

        if not self.training:
            value = self.inverse_value_transform(value)

        return NetworkOutput(value, 0, actor_logit, state)

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        state, reward = self.dynamics(hidden_state, action)
        actor_logit, value = self.prediction(state)

        if not self.training:
            value = self.inverse_value_transform(value)
            reward = self.inverse_reward_transform(reward)

        return NetworkOutput(value, reward, actor_logit, state)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

# MuZero Network Structure with Fully Connected layers
class FullyConnectedNetwork(Network):
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.config = config
        self.prediction_policy_network = nn.Sequential()
        self.prediction_value_network = nn.Sequential()
        self.representation_network = nn.Sequential()
        self.dynamics_state_network = nn.Sequential()
        self.dynamics_reward_network = nn.Sequential()

    def prediction(self, hidden_state):
        policy_logits = self.prediction_policy_network(hidden_state)
        value = self.prediction_value_network(hidden_state)
        return policy_logits, value

    def representation(self, image):
        return self.representation_network(image)

    def dynamics(self, hidden_state, action):
        one_hot = torch.zeros(size=(action.shape[0], self.config.action_space_size),
                              dtype=torch.float32, device=action.device)
        one_hot.scatter(1, action, 1.0)
        x = torch.cat((hidden_state, one_hot), dim=1)

        next_state = self.dynamics_state_network(x)
        reward = self.dynamics_reward_network(x)

        return next_state, reward


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
"""

# MuZero Network Structure with Residual Blocks
# class ResidualNetwork(Network):
#     def __init__(self, num_channels, num_blocks):
#         super().__init__()
#         self.prediction_policy_network =
#         self.prediction_value_network =
#         self.representation_network =
#         self.dynamics_state_network =
#         self.dynamics_reward_network =
#
#     def prediction(self, hidden_state):
#         policy_logits = self.prediction_policy_network(hidden_state)
#         value = self.prediction_value_network(hidden_state)
#         return policy_logits, value
#
#     def representation(self, image):
#
#     def dynamics(self, hidden_state, action):
#

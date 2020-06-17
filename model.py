import typing
from typing import Dict, List

import torch.nn as nn
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[int, float]
    hidden_state: List[float]

class UniformNetwork():
    """policy -> uniform, value -> 0, reward -> 0"""

    def __init__(self, action_size):
        self.action_size = action_size

    def initial_inference(self, image):
        return NetworkOutput(None, None, [[1 / self.action_size for i in range(self.action_size)]], None)

    def recurrent_inference(self, hidden_state, action):
        return NetworkOutput(None, None, [[1 / self.action_size for i in range(self.action_size)]], None)


class Network(ABC, nn.Module):
    def __init__(self, scaling=False, down_sample=False):
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

    def initial_inference(self, image):
        hidden_state = self.representation(image)
        policy_logits, value = self.prediction(hidden_state)

        return NetworkOutput(value, None, policy_logits, hidden_state)

    def recurrent_inference(self, hidden_state, action):
        next_state, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_state)

        return NetworkOutput(value, reward, policy_logits, next_state)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

class MuZeroNetwork(Network):
    def __init__(self, config):
        super().__init__()
        self.action_space_size = config.action_space_size
        self.full_support_size = 2 * config.support_size + 1

        self.representation_network = FCN(config.observation_shape[2], config.fc_representation_layers, config.encoding_size,)

        self.dynamics_encoded_state_network = FCN(config.encoding_size + self.action_space_size, config.fc_dynamics_layers, config.encoding_size)
        self.dynamics_reward_network = FCN(config.encoding_size, config.fc_reward_layers, self.full_support_size,)

        self.prediction_policy_network = FCN(config.encoding_size, config.fc_policy_layers, self.action_space_size)
        self.prediction_value_network = FCN(config.encoding_size, config.fc_value_layers, self.full_support_size,)

    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation.view(observation.shape[0], -1))
        return self.normalize_encoded_state(encoded_state)

    def dynamics(self, encoded_state, action):
        action_one_hot = torch.zeros((action.shape[0], self.action_space_size)).to(action.device).float()
        
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)
        reward = self.dynamics_reward_network(next_encoded_state)

        return self.normalize_encoded_state(next_encoded_state), reward

    def normalize_encoded_state(self, encoded_state):
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        return (encoded_state - min_encoded_state) / scale_encoded_state


class FCN(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super().__init__()
        size_list = [input_size] + layer_sizes
        layers = []
        if 1 < len(size_list):
            for i in range(len(size_list) - 1):
                layers.extend(
                    [
                        torch.nn.Linear(size_list[i], size_list[i + 1]),
                        torch.nn.LeakyReLU(),
                    ]
                )
        layers.append(torch.nn.Linear(size_list[-1], output_size))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    if logits is None:
        return 0

    probabilities = torch.softmax(logits, dim=1)
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x.item()


def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits
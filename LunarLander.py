from typing import List

import gym
from game_env import Game
import numpy as np
from config import MuZeroConfig

class LunarLander(Game):
    def __init__(self, action_space_size, discount):
        super().__init__(action_space_size, discount)
        self.env = gym.make("LunarLander-v2")
        self.done = False

    def step(self, action):
        observation, reward, done, _ = self.env.step(action)
        self.done = done
        return np.array([[observation]]), reward/3

    def terminal(self):
        """Is the game is finished?"""
        return self.done

    def legal_actions(self):
        """Return the legal actions available at this instant."""
        return [i for i in range(4)]

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return np.array([[self.env.reset()]])

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

def make_lunarlander_config():
    def visit_softmax_temperature(num_moves, training_steps):
        if num_moves < 0.5 * training_steps:
            return 1.0
        elif num_moves < 0.75 * training_steps:
            return 0.5
        else:
            return 0.25

    return MuZeroConfig(
        game=LunarLander,
        action_space_size=4,
        max_moves=500,
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=50,
        num_training_loop=50,
        num_epochs=200000,
        batch_size=32,
        td_steps=50,
        num_train_episodes=30,
        num_eval_episodes=10,
        lr_init=0.05,
        lr_decay_steps=1000,
        max_priority=False,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        network_args={'support_size': 10,
                      'encoding_size': 10,
                      'rep_hidden': [],
                      'dyn_hidden': [64],
                      'rew_hidden': [64],
                      'val_hidden': [64],
                      'pol_hidden': [],
                      'observation_shape': (1, 1, 8),
                      },
        result_path="lunarlander.weights"
        )
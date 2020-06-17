from typing import List

import gym
from game_env import Game
import numpy as np
from config import MuZeroConfig

class CartPole(Game):
    def __init__(self, action_space_size, discount):
        super().__init__(action_space_size, discount)
        self.env = gym.make('CartPole-v1')
        self.done = False

    def step(self, action):
        """Execute one step of the game conditioned by the given action."""
        observation, reward, done, _ = self.env.step(action)
        self.done = done
        return np.array([[observation]]), reward

    def terminal(self):
        """Is the game is finished?"""
        return self.done

    def legal_actions(self):
        """Return the legal actions available at this instant."""
        return [i for i in range(2)]

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

def make_cartpole_config():
    def visit_softmax_temperature(num_moves, training_steps):
        return 1.0

    return MuZeroConfig(
        game=CartPole,
        action_space_size=2,
        max_moves=1000,
        discount=0.99,
        dirichlet_alpha=0.25,
        num_simulations=11,
        num_training_loop=50,
        num_epochs=20,
        batch_size=128,
        td_steps=10,
        num_train_episodes=20,
        num_eval_episodes=1,
        lr_init=0.05,
        lr_decay_steps=1000,
        max_priority=False,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        network_args={'support_size': 10,
                      'encoding_size': 8,
                      'rep_hidden': [],
                      'dyn_hidden': [16],
                      'rew_hidden': [16],
                      'val_hidden': [],
                      'pol_hidden': [],
                      'observation_shape': (1, 1, 4),
                      },
        result_path="cartpole.weights"
        )
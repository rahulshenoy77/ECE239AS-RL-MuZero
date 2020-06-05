from typing import List

import gym

from game_env import Action, Game
import numpy as np
from config import MuZeroConfig
from model import Network, UniformNetwork


class ScalingObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that apply a min-max scaling of observations.
    """
    def __init__(self, env, low=None, high=None):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)

        low = np.array(self.observation_space.low if low is None else low)
        high = np.array(self.observation_space.high if high is None else high)

        self.mean = (high + low) / 2
        self.max = high - self.mean

    def observation(self, observation):
        return (observation - self.mean) / self.max

class CartPole(Game):
    def __init__(self, action_space_size, discount):
        super().__init__(action_space_size, discount)
        self.env = gym.make('CartPole-v1')
        self.env = ScalingObservationWrapper(self.env, low=[-2.4, -2.0, -0.42, -3.5], high=[2.4, 2.0, 0.42, 3.5])
        self.actions = list(map(lambda i: Action(i), range(self.env.action_space.n)))
        self.observations = [self.env.reset()]
        self.done = False

    def step(self, action):
        """Execute one step of the game conditioned by the given action."""
        observation, reward, done, _ = self.env.step(action.index)
        self.observations += [observation]
        self.done = done
        return reward

    def terminal(self):
        """Is the game is finished?"""
        return self.done

    def legal_actions(self):
        """Return the legal actions available at this instant."""
        return self.actions

    def make_image(self, state_index):
        """Compute the state of the game."""
        return self.observations[state_index]
"""
class CartPoleNetwork(Network):
    def __init__(self, input_size, action_space_n, reward_support_size, value_support_size,
                 inverse_value_transform, inverse_reward_transform):
        super(Network, self).__init__(inverse_value_transform, inverse_reward_transform)
        self.hx_size = 32
        self._representation = nn.Sequential(nn.Linear(input_size, self.hx_size),
                                             nn.Tanh())
        self._dynamics_state = nn.Sequential(nn.Linear(self.hx_size + action_space_n, 64),
                                             nn.Tanh(),
                                             nn.Linear(64, self.hx_size),
                                             nn.Tanh())
        self._dynamics_reward = nn.Sequential(nn.Linear(self.hx_size + action_space_n, 64),
                                              nn.LeakyReLU(),
                                              nn.Linear(64, reward_support_size))
        self._prediction_actor = nn.Sequential(nn.Linear(self.hx_size, 64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64, action_space_n))
        self._prediction_value = nn.Sequential(nn.Linear(self.hx_size, 64),
                                               nn.LeakyReLU(),
                                               nn.Linear(64, value_support_size))
        self.action_space_n = action_space_n

        self._prediction_value[-1].weight.data.fill_(0)
        self._prediction_value[-1].bias.data.fill_(0)
        self._dynamics_reward[-1].weight.data.fill_(0)
        self._dynamics_reward[-1].bias.data.fill_(0)

    def prediction(self, state):
        actor_logit = self._prediction_actor(state)
        value = self._prediction_value(state)
        return actor_logit, value

    def representation(self, obs_history):
        return self._representation(obs_history)

    def dynamics(self, state, action):
        assert len(state.shape) == 2
        assert action.shape[1] == 1

        action_one_hot = torch.zeros(size=(action.shape[0], self.action_space_n),
                                     dtype=torch.float32, device=action.device)
        action_one_hot.scatter_(1, action, 1.0)

        x = torch.cat((state, action_one_hot), dim=1)
        next_state = self._dynamics_state(x)
        reward = self._dynamics_reward(x)
        return next_state, reward
"""
def make_cartpole_config():
    def visit_softmax_temperature(num_moves, training_steps):
        return 1.0

    return MuZeroConfig(
        game=CartPole,
        action_space_size=2,
        max_moves=1000,
        discount=0.99,
        dirichlet_alpha=0.25,
        num_simulations=11,  # Odd number perform better in eval mode
        num_training_loop=50,
        batch_size=512,
        td_steps=10,
        num_train_episodes=20,
        lr_init=0.05,
        lr_decay_steps=1000,
        max_priority=False,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        #nb_epochs=20,
        network_args={'action_size': 2,
                      'state_size': 4,
                      'representation_size': 4,
                      'max_value': 500},
        #network=CartPoleNetwork,
        network=UniformNetwork,
        )
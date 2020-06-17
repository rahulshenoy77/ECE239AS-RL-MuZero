from typing import Optional
import collections
from model import MuZeroNetwork

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])

class MuZeroConfig(object):
    def __init__(self,
                 game,
                 action_space_size: int,
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 num_training_loop: int,
                 num_epochs,
                 batch_size: int,
                 td_steps: int,
                 num_train_episodes,
                 num_eval_episodes,
                 lr_init: float,
                 lr_decay_steps: float,
                 max_priority: bool,
                 visit_softmax_temperature_fn,
                 network_args,
                 result_path,
                 known_bounds: Optional[KnownBounds] = None,
                 ):

        ### Self-Play
        self.game = game
        self.action_space_size = action_space_size
        self.num_train_episodes = num_train_episodes
        self.eval_episodes = num_eval_episodes

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.num_training_loop = num_training_loop
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        ### Training
        self.training_steps = int(1000e3)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = batch_size
        self.num_unroll_steps = 10
        self.td_steps = td_steps
        self.max_priority = max_priority

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps

        self.result_path = result_path
        self.device = "cuda"
        self.step_counter = 0

        self.support_size = network_args["support_size"]
        self.encoding_size = network_args["encoding_size"]
        self.fc_representation_layers = network_args["rep_hidden"]
        self.fc_dynamics_layers = network_args["dyn_hidden"]
        self.fc_reward_layers = network_args["rew_hidden"] 
        self.fc_value_layers = network_args["val_hidden"] 
        self.fc_policy_layers = network_args["pol_hidden"]
        self.observation_shape = network_args["observation_shape"] 
        self.action_space = [i for i in range(self.action_space_size)]
        self.players = [i for i in range(1)]

    def new_game(self):
        return self.game(self.action_space_size, self.discount)

    def new_network(self):
        return MuZeroNetwork(self)

    def incr_counter(self):
        self.step_counter += 1

    def get_counter(self):
        return self.step_counter
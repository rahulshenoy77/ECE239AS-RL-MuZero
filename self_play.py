from mcts import run_mcts, select_action, expand_node, add_exploration_noise, Node
import numpy as np
import torch

def run_selfplay(config, storage, replay_buffer):
    network = storage.latest_network()
    returns = []
    for _ in range(config.num_train_episodes):
        with torch.no_grad():
            game_history = play_game(config, network, train=True)
        replay_buffer.save_game(game_history)
        returns.append(sum(game_history.reward_history))
    return sum(returns) / config.num_train_episodes


def run_eval(config, storage):
    network = storage.latest_network()
    returns = []
    for _ in range(config.eval_episodes):
        with torch.no_grad():
            game_history = play_game(config, network, train=False)
            returns.append(sum(game_history.reward_history))
    return sum(returns) / config.eval_episodes if config.eval_episodes else 0


def play_game(config, network, train):
    """
    Each game is produced by starting at the initial board position, then
    repeatedly executing a Monte Carlo Tree Search to generate moves until the end
    of the game is reached.
    """
    game = config.new_game()

    game_history = GameHistory()
    observation = game.reset()
    game_history.apply(0, observation, 0)

    while not game.terminal() and len(game_history.action_history) < config.max_moves:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game_history.make_image(-1)
        current_observation = torch.tensor(observation).float().unsqueeze(0)

        expand_node(config, root, game.to_play(), game.legal_actions(), network.initial_inference(current_observation))
        if train:
            add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the networks.
        run_mcts(config, root, game, network)
        action = select_action(config, len(game_history.action_history), root, train)

        observation, reward = game.step(action)
        game_history.store_search_statistics(root, config.action_space)
        game_history.apply(action, observation, reward)

    game.close()

    return game_history

class GameHistory:
    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.child_visits = []
        self.root_values = []
        self.priorities = None

    def store_search_statistics(self, root, action_space):
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            self.child_visits.append([
                root.children[a].visit_count / sum_visits if a in root.children else 0
                for a in action_space
            ])
            self.root_values.append(root.value())
        else:
            self.root_values.append(None)

    def apply(self, action, obs, reward):
        self.action_history.append(action)
        self.observation_history.append(obs)
        self.reward_history.append(reward)

    def make_image(self, state_index):
        return self.observation_history[state_index]
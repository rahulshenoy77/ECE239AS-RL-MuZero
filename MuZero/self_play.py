from mcts import run_mcts, select_action, expand_node, add_exploration_noise, Node


def run_selfplay(config, storage, replay_buffer):
    """Take the latest network, produces multiple games and save them in the shared replay buffer"""
    network = storage.latest_network()
    returns = []
    for _ in range(config.num_train_episodes):
        game = play_game(config, network, train=True)
        replay_buffer.save_game(game)
        returns.append(sum(game.rewards))
    return sum(returns) / config.num_train_episodes


def run_eval(config, storage):
    """Evaluate MuZero without noise added to the prior of the root and without softmax action selection"""
    network = storage.latest_network()
    returns = []
    for _ in range(config.eval_episodes):
        game = play_game(config, network, train=False)
        returns.append(sum(game.rewards))
    return sum(returns) / config.eval_episodes if config.eval_episodes else 0


def play_game(config, network, train):
    """
    Each game is produced by starting at the initial board position, then
    repeatedly executing a Monte Carlo Tree Search to generate moves until the end
    of the game is reached.
    """
    game = config.new_game()

    while not game.terminal() and len(game.history) < config.max_moves:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1)
        expand_node(root, game.to_play(), game.legal_actions(), network.initial_inference(current_observation))
        if train:
            add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the networks.
        run_mcts(config, root, game.action_history(), network)
        action = select_action(config, len(game.history), root, network, train)
        game.apply(action)
        game.store_search_statistics(root)
    return game
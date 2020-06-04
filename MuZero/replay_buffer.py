from .config import MuZeroConfig
import numpy as np


class ReplayBuffer(object):

    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []
        self.game_priorities = []
        self.pos_priorities = []
        # Whether to assign current max priority to new game history
        self.max_priority = config.max_priority

    def save_game(self, game, priorities=None):
        if priorities is None:
            if self.max_priority:
                max_priority = max(self.game_priorities)
                priorities = np.array([max_priority for i in range(len(game))])
            else:
                # if not what to do, calculate or just assign an instant??
                priorities = np.array([1 for i in range(len(game))])

        if len(self.buffer) > self.window_size:
            # delete the oldest game history ???
            # maybe consider changing to least priority
            self.buffer.pop(0)
            self.pos_priorities.pop(0)
            self.game_priorities.pop(0)

        self.buffer.append(game)
        self.pos_priorities.append(priorities)
        # use which standard for game priority ??? mean/max/weighted mean??
        self.game_priorities.append(np.max(priorities))

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        game_indices = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(self.buffer[idx], self.sample_position(idx)) for idx in game_indices]

        return [(g.make_image(i),
                 g.history[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps, g.to_play()))
                for (g, i) in game_pos]

    def sample_game(self) -> int:
        # Sample game from buffer either uniformly or according to some priority.
        prob = np.array(self.game_priorities)
        prob /= np.sum(prob)
        idx = np.random.choice(len(self.buffer), p=prob)

        return idx

    def sample_position(self, idx) -> int:
        # Sample position from game either uniformly or according to some priority.
        priorities = self.pos_priorities[idx]
        prob = priorities / np.sum(priorities)
        position = np.random.choice(len(priorities), p=prob)

        return position

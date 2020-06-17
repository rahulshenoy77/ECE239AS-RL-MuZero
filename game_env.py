from typing import List

class Player(object):
    """
    A one player class.
    This class is useless, it's here for legacy purpose and for potential adaptations for a two players MuZero.
    """

    def __eq__(self, other):
        return True

class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(self, action_space_size, discount):
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.discount = discount
        self.action_space_size = action_space_size
    
    def to_play(self) -> Player:
        return Player()
    
    def __len__(self):
        return len(self.rewards)

    # inheritence
    def step(self, action):
        # return reward
        pass

    def terminal(self):
        # Game specific termination rules.
        pass

    def legal_actions(self):
        # Game specific calculation of legal actions.
        return []

    def make_image(self, state_index):
        # Game specific feature planes.
        return []
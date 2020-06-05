from typing import List


class Action(object):
    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index

class Player(object):
    """
    A one player class.
    This class is useless, it's here for legacy purpose and for potential adaptations for a two players MuZero.
    """

    def __eq__(self, other):
        return True

class ActionHistory(object):
    """
    Simple history container used inside the search.
    Only used to keep track of the actions executed.
    """

    def __init__(self, history, action_space_size):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action):
        self.history.append(action)

    def last_action(self):
        return self.history[-1]

    def action_space(self):
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self):
        return Player()

class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(self, action_space_size, discount):
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.discount = discount
        self.action_space_size = action_space_size

    def apply(self, action):
        reward = self.step(action)
        self.rewards.append(reward)
        self.history.append(action)

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())

    def make_target(self, state_index, num_unroll_steps, td_steps, to_play):
        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount**td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount ** i

            if current_index < len(self.root_values):
                targets.append((value, self.rewards[current_index],
                                self.child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, 0, []))
        return targets

    def to_play(self) -> Player:
        return Player()

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)

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
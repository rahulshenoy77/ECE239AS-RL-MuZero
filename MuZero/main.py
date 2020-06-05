import os
from config import *
from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer
from self_play import run_selfplay
from CartPole import make_cartpole_config

class MuZero:
    def __init__(self, config):
        self.config = config

    def train(self):
        #os.makedirs(self.config.results_path, exist_ok=True)

        storage = SharedStorage(self.config.new_network(), self.config.get_uniform_network())
        replay_buffer = ReplayBuffer(self.config)

        for loop in range(self.config.num_training_loop):
            print(f"Loop: {loop}")
            score_train = run_selfplay(self.config, storage, replay_buffer)
            #train_network(self.config, storage, replay_buffer, self.config.nb_epochs)

        return storage.latest_network()

if __name__ == '__main__':
    a = MuZero(make_cartpole_config())
    a.train()
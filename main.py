import os
from config import *
from shared_storage import SharedStorage
from self_play import run_selfplay, run_eval
from CartPole import make_cartpole_config
from LunarLander import make_lunarlander_config
from FrozenLake import make_frozenlake_config
import torch

class MuZero:
    def __init__(self, config):
        self.config = config

    def test(self):
        
        model = self.config.new_network()
        model.set_weights(torch.load(self.config.result_path))

        storage = SharedStorage(model, model)
        score = run_eval(self.config, storage)
        print(f"Test score: {score}")

if __name__ == '__main__':
    task = {
        "0": make_cartpole_config,
        "1": make_lunarlander_config,
        "2": make_frozenlake_config,
    }
    
    choice = input("choose a task (0: CartPole, 1: LunarLander, 2: Frozenlake): ")
    alg = MuZero(task[choice]())
    alg.test()

    #except:
    #    print("invalid input")
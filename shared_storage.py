class SharedStorage(object):

    def __init__(self, network, uniform_network):
        self._networks = {}
        self.current_network = network
        self.uniform_network = uniform_network
        self.step_counter = 0

    def latest_network(self):
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return self.uniform_network

    def save_network(self, step, network):
        self._networks[step] = network

    
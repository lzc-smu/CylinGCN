from .trainer import Trainer
from .criterion import NetworkWrapper


def make_trainer(network):
    network = NetworkWrapper(network)
    return Trainer(network)

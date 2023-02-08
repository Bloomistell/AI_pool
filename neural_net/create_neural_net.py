import torch
from torch import nn


class DQN(nn.Module):

    def __init__(self, n_observations, layers, n_actions):
        super(DQN, self).__init__()

        layer_list = [nn.Linear(n_observations, layers[0])]
        for i in range(len(layers)-1):
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Linear(layers[i], layers[i+1]))

        layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(layers[-1], n_actions))

        self.network = nn.Sequential(layer_list)

    def forward(self, x):
        return self.network(x)
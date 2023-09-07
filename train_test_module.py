import torch
import torch.nn as nn
import numpy as np
from topology import Topology, map_int_to_optim, map_int_to_activation
from sklearn.metrics import accuracy_score


class NeuralNet(nn.Module):
    def __init__(self, topology, input_channels_number, num_of_options):
        super(NeuralNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_channels_number, out_features=topology.get_l1_neurons()),
            map_int_to_activation(topology.get_l1_activation()),
            nn.Linear(in_features=topology.get_l1_neurons(), out_features=topology.get_l2_neurons()),
            map_int_to_activation(topology.get_l2_activation()),
            nn.Linear(in_features=topology.get_l2_neurons(), out_features=num_of_options)
        )

    def forward(self, x):
        return self.classifier(x)


def get_trained_model(topology, x_train, y_train):
    if topology.get_fitness() is not None:
        return None
    n_features = len(x_train[0])

    input_size = n_features
    output_size = len(np.unique(y_train))

    model = NeuralNet(topology, input_size, output_size)

    lr = topology.get_lr()

    criterion = nn.CrossEntropyLoss()

    optimizer = map_int_to_optim(model, lr, topology.get_optim())
    num_epochs = 1000

    for epoch in range(num_epochs):
        y_predicted = model(x_train)
        loss = criterion(y_predicted, y_train)

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

    return model


def get_performance(model, x_test, y_test):
    predicted = model(x_test).detach()
    _, maxxed = torch.max(predicted, 1)
    return accuracy_score(y_test, maxxed)

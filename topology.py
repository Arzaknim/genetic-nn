import numpy as np
import torch
import torch.nn as nn


class Topology:

    def __init__(self, params=None):
        if params:
            self.lr = params[0]
            self.optim = params[1]
            self.l1_output = params[2]
            self.l2_output = params[3]
            self.l1_activation = params[4]
            self.l2_activation = params[5]

        else:
            self.lr = random_float(0, 0.1)
            self.optim = random_int(0, 9)
            self.l1_output = random_int(25, 51)
            self.l2_output = random_int(5, 25)
            self.l1_activation = random_int(0, 11)
            self.l2_activation = random_int(0, 11)
        self.fitness = None

    def get_genome(self):
        return [self.lr, self.optim, self.l1_output, self.l2_output,
                self.l1_activation, self.l2_activation]

    def set_fitness(self, number):
        if not self.fitness:
            self.fitness = number

    def get_fitness(self):
        return self.fitness

    def get_l1_neurons(self):
        return self.l1_output

    def get_l2_neurons(self):
        return self.l2_output

    def get_lr(self):
        return self.lr

    def get_epochs(self):
        return self.epochs

    def get_l1_activation(self):
        return self.l1_activation

    def get_l2_activation(self):
        return self.l2_activation

    def get_optim(self):
        return self.optim


def random_float(low, upper):
    return np.random.uniform(low, upper)


def random_int(low, upper):
    return np.random.randint(low, upper)


def map_int_to_activation(integer):
    fn = None
    if integer == 0:
        fn = nn.ReLU()
    if integer == 1:
        fn = nn.Sigmoid()
    if integer == 2:
        fn = nn.ELU()
    if integer == 3:
        fn = nn.CELU()
    if integer == 4:
        fn = nn.Tanh()
    if integer == 5:
        fn = nn.GELU()
    if integer == 6:
        fn = nn.RReLU()
    if integer == 7:
        fn = nn.LeakyReLU()
    if integer == 8:
        fn = nn.ReLU6()
    if integer == 9:
        fn = nn.PReLU()
    if integer == 10:
        fn = nn.LogSigmoid()

    return fn


def map_int_to_optim(model, lr, integer):
    fn = None
    if integer == 0:
        fn = torch.optim.ASGD(model.parameters(), lr=lr)
    if integer == 1:
        fn = torch.optim.SGD(model.parameters(), lr=lr)
    if integer == 2:
        fn = torch.optim.Adam(model.parameters(), lr=lr)
    if integer == 3:
        fn = torch.optim.Adagrad(model.parameters(), lr=lr)
    if integer == 4:
        fn = torch.optim.Rprop(model.parameters(), lr=lr)
    if integer == 5:
        fn = torch.optim.RMSprop(model.parameters(), lr=lr)
    if integer == 6:
        fn = torch.optim.Adamax(model.parameters(), lr=lr)
    if integer == 7:
        fn = torch.optim.Adadelta(model.parameters(), lr=lr)
    if integer == 8:
        fn = torch.optim.RAdam(model.parameters(), lr=lr)

    return fn

import math
import random as rnd
from topology import Topology, random_float, random_int
from train_test_module import get_performance, get_trained_model


class GeneticAlgorithm:

    def __init__(self, algo_params, X_train, X_test, y_train, y_test):
        self.num_topologies = algo_params['num_topologies']
        self.generations = algo_params['generations']
        self.mutation = algo_params['mutation']
        self.population = self.init_population()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def run(self):
        for idx in range(len(self.population)):
            topology = self.population[idx]
            model = get_trained_model(topology, self.X_train, self.y_train)
            if model is not None:
                topology.set_fitness(get_performance(model, self.X_test, self.y_test))

        for i in range(self.generations):
            self.keep_top_half()
            self.multiply()
            for idx in range(len(self.population)):
                topology = self.population[idx]
                model = get_trained_model(topology, self.X_train, self.y_train)
                if model is not None:
                    topology.set_fitness(get_performance(model, self.X_test, self.y_test))
            self.population.sort(key=lambda x: x.get_fitness(), reverse=True)
            if (i + 1) % 10 == 0:
                print(f'Best fitness of generation {i+1} out of {self.generations}:')
                print(self.population[0].get_fitness())
                print('lr, optim, l1 neurons, l2 neurons, l1 activation. l2 activation')
                print(self.population[0].get_genome())
                print('--------------')

    def init_population(self):
        result = []
        for i in range(self.num_topologies):
            result.append(Topology())

        return result

    def keep_top_half(self):
        limit = math.floor(self.num_topologies/2)
        self.population.sort(key=lambda x: x.get_fitness(), reverse=True)
        self.population = self.population[:limit]

    def single_point_crossover(self, parent1, parent2):
        genome1 = parent1.get_genome()
        genome2 = parent2.get_genome()
        upper = 6
        crossover_point = rnd.randint(1, upper)
        params = []
        idx = 0
        while idx < crossover_point:
            params.append(genome1[idx])
            idx += 1

        while idx < upper:
            params.append(genome2[idx])
            idx += 1

        return Topology(params)

    def mutate(self, individual):
        genome = individual.get_genome()
        params = []
        changed = 0
        for i in range(len(genome)):
            p = rnd.random()
            if p < self.mutation:
                changed += 1
                if i == 0:
                    params.append(random_float(0, 0.1))
                elif i == 1:
                    params.append(random_int(0, 9))
                elif i in [2, 3]:
                    params.append(random_int(1, 51))
                elif i in [4, 5]:
                    params.append(random_int(0, 11))

            else:
                params.append(genome[i])

        if changed:
            return Topology(params)
        return individual

    def multiply(self):
        result = []
        size = self.num_topologies - len(self.population)
        while len(result) != size:
            parent_a = rnd.choice(self.population)
            parent_b = rnd.choice(self.population)
            while parent_a is parent_b:
                parent_a = rnd.choice(self.population)
                parent_b = rnd.choice(self.population)

            child = self.single_point_crossover(parent_a, parent_b)
            child = self.mutate(child)
            result.append(child)

        self.population = self.population + result

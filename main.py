from data_preparation import prepare_data
from genetic_algorithm import GeneticAlgorithm

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = prepare_data('data.csv')

    algo_params = {
        'num_topologies': 10,
        'generations': 50,
        'mutation': 0.05
    }

    GeneticAlgorithm(algo_params, X_train, X_test, y_train, y_test).run()

from random import shuffle
from time import time
from typing import List

from matplotlib import pyplot as plt

from genetic_algo import GeneticAlgorithm
from utils import generate_adjacency_matrix


def generate_population(population_size: int, n_cities: int) -> List[List[int]]:
    population = []
    for _ in range(population_size):
        chromosome = list(range(1, n_cities))
        shuffle(chromosome)
        chromosome = [0] + chromosome
        population.append(chromosome)
    return population


def calc_cost(adjacency_matrix, chromosome):
    cost = 0
    for i in range(1, len(chromosome)):
        u = chromosome[i - 1]
        v = chromosome[i]
        cost += adjacency_matrix[u][v]
    return cost


def main():
    sssc_ga = GeneticAlgorithm(crossover_type="SSSC")
    essc_ga = GeneticAlgorithm(crossover_type="ESSC")

    sssc_ga_times = []
    essc_ga_times = []

    test_range = range(20, 41, 2)

    for size in test_range:
        adjacency_matrix = generate_adjacency_matrix(size)

        def fitness_function(chromosomes: List[int]):
            _chromosomes = chromosomes[:]
            _chromosomes.append(_chromosomes[0])
            return 1 / calc_cost(adjacency_matrix, _chromosomes)

        initial_population = generate_population(10, size)
        sssc_ga_start_time = time()
        sssc_ga_best_solution = sssc_ga.evalute(
            initial_population,
            mutation_probabilty=0.1,
            fitness_func=fitness_function,
            max_iter=size * 50,
        )
        sssc_ga_end_time = time()
        sssc_ga_times.append(sssc_ga_end_time - sssc_ga_start_time)

        initial_population = generate_population(10, size)
        essc_ga_start_time = time()
        essc_ga_best_solution = essc_ga.evalute(
            initial_population,
            mutation_probabilty=0.1,
            fitness_func=fitness_function,
            max_iter=size * 50,
        )
        essc_ga_end_time = time()
        essc_ga_times.append(essc_ga_end_time - essc_ga_start_time)

    _, axes = plt.subplots(2, 2)
    axes[0][0].plot(test_range, sssc_ga_times, label="SSSC GA Time plot")
    axes[0][0].plot(test_range, essc_ga_times, label="ESSC GA Time plot")
    axes[0][0].set_xlabel("Number of cities")
    axes[0][0].set_ylabel("Time taken (ns)")
    axes[0][0].legend()

    axes[0][1].plot(sssc_ga.inverse_max_history, label="SSSC GA min cost")
    axes[0][1].plot(essc_ga.inverse_max_history, label="ESSC GA min cost")
    axes[0][1].set_ylabel("Minimum travelling cost")
    axes[0][1].set_xlabel("Number of iterations")
    axes[0][1].title.set_text("For last set of cities")
    axes[0][1].legend()

    axes[1][0].plot(sssc_ga.max_history, label="SSSC GA maximum fitness")
    axes[1][0].plot(essc_ga.max_history, label="ESSC GA maximum fitness")
    axes[1][0].set_ylabel("Maximum fitness")
    axes[1][0].set_xlabel("Number of iterations")
    axes[1][0].title.set_text("For last set of cities")
    axes[1][0].legend()

    axes[1][1].plot(sssc_ga.mean_history, label="SSSC GA average fitness")
    axes[1][1].plot(essc_ga.mean_history, label="ESSC GA average fitness")
    axes[1][1].set_ylabel("Average fitness")
    axes[1][1].set_xlabel("Number of iterations")
    axes[1][1].title.set_text("For last set of cities")
    axes[1][1].legend()

    plt.show()


if __name__ == "__main__":
    main()

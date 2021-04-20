import random
from statistics import mean
from types import FunctionType
from typing import List, Literal, Tuple, Union


class GeneticAlgorithm:
    """Implementation of genetic algorithm."""

    def __init__(
        self, crossover_type: Literal["Binary", "SSSC", "ESSC"] = "Binary"
    ) -> None:
        """
        Constructor of GeneticAlgorithm.

        Arguments:
            crossover_type (literal): Type of crossover function to be used.
                Default = "Binary"
        """
        self.__crossover_type = crossover_type

    def __population_fitness(
        self, population: List[Union[List[int], str]]
    ) -> List[float]:
        """
        Computes fitness value in percentage for the population.

        Arguments:
            population (list): population of chromosomes.

        Returns:
            (list): fitness value with population corresponding index.
        """
        ## self.__population_fitness_cache is used to reduce time required in computation.
        population_fitness = self.__population_fitness_cache
        for chromosome in population[len(population_fitness) :]:
            pro = self.__fitness_func(chromosome)
            population_fitness.append(pro)
        self.__population_fitness_cache = population_fitness
        return population_fitness

    def select(
        self,
        population: List[Union[List[int], str]],
        population_fitness: List[float],
        n_parents: int,
    ) -> Union[List[List[int]], List[int], List[str]]:
        """
        Select 2 parents from given population with preference to the individuals with good fitness scores and allow them to pass there genes to the successive generations.

        Arguments:
            population (list): Population of chromosomes.
            n_parent (int): Number of parents to be selected.

        Returns:
            (list): n_parents parents [parent1, parent2, parentn].
        """
        parents = random.choices(population, population_fitness, k=n_parents)
        return parents

    def __binary_crossover(
        self, parents: Tuple[List[int], str]
    ) -> Union[List[int], str]:
        chromosome_len = len(parents[0])
        children = [[-1 for _ in range(chromosome_len)] for _ in range(2)]

        points = random.sample(range(0, chromosome_len), 2)

        start_point = min(points)
        end_point = max(points)

        children[0][start_point:end_point] = parents[0][start_point:end_point]
        children[1][start_point:end_point] = parents[1][start_point:end_point]

        child_index = [0, 0]

        ## fill remaining cities
        for i in range(0, chromosome_len):

            if parents[1][i] not in children[0]:
                while children[0][child_index[0]] != -1:
                    child_index[0] += 1
                children[0][child_index[0]] = parents[1][i]

            if parents[0][i] not in children[1]:
                while children[1][child_index[1]] != -1:
                    child_index[1] += 1
                children[1][child_index[1]] = parents[0][i]

        ## return child which produces better fitness
        child0_fitness = self.__fitness_func(children[0])
        child1_fitness = self.__fitness_func(children[1])

        return children[0] if child0_fitness > child1_fitness else children[1]

    def __real_swap(self, parent):
        chromosome_len = len(parent)
        # pivot = (
        #     chromosome_len // 2
        # )  ## hardcoded for better result for given small number of cities (in 10 - 20 range).

        starts = sorted(random.sample(range(1, chromosome_len), k=2))
        length = random.randint(
            0, min(chromosome_len - starts[1], starts[1] - starts[0])
        )

        child = parent[:]
        (
            child[starts[1] : starts[1] + length],
            child[starts[0] : starts[0] + length],
        ) = (
            child[starts[0] : starts[0] + length],
            child[starts[1] : starts[1] + length],
        )

        return child

    def __sss_crossover(
        self, parents: Union[List[int], str]
    ) -> Union[List[int], str]:
        if len(parents) != 1:
            raise ("Must have one parent only")
        parent = parents[0]
        child = self.__real_swap(parent)

        return child

    def __ess_crossover(
        self, parents: Union[List[int], str]
    ) -> Union[List[int], str]:
        if len(parents) != 1:
            raise ("Must have one parent only")
        parent = parents[0]
        child = self.__real_swap(parent)

        parent_fitness = self.__fitness_func(parent)
        child_fitness = self.__fitness_func(child)

        return child if child_fitness > parent_fitness else parent

    def crossover(
        self, parents: Tuple[List[int], str]
    ) -> Union[List[int], str]:
        """
        Mating between individuals. Two individuals are selected using selection operator and crossover sites are chosen randomly. Then the genes at these crossover sites are exchanged thus creating a completely new individual (child).

        Arguments:
            parents (tuples): Mating parents chromosome.

        Returns:
            (str or list): Child chromosome.
        """
        if len(parents) == 0:
            raise ("Excepting more parents")

        if self.__crossover_type == "Binary":
            return self.__binary_crossover(parents)
        elif self.__crossover_type == "SSSC":
            return self.__sss_crossover(parents)
        elif self.__crossover_type == "ESSC":
            return self.__ess_crossover(parents)

    def mutate(
        self,
        child: Union[List[int], str],
        mutation_probablity: int,
    ) -> Union[List[int], str]:
        """
        Mutate to insert random genes in offspring to maintain the diversity in population to avoid the premature convergence.

        Arguments:
            child (str or list): Child's chromosome.
            mutation_probablity (int): Chance for each gene that mutation takes place.

        Returns:
            (str or list): Mutated child's chromosome.
        """
        if child == None:
            return None
        mutation_child = child
        chromosome_len = len(child)
        for _ in range(chromosome_len):
            if random.random() < mutation_probablity:
                points = random.sample(range(0, chromosome_len), 2)
                mutation_child[points[0]], mutation_child[points[1]] = (
                    mutation_child[points[1]],
                    mutation_child[points[0]],
                )

        return mutation_child

    def evalute(
        self,
        initial_population: List[Union[List[int], str]],
        mutation_probabilty: float,
        fitness_func: FunctionType,
        max_iter: int = 1000,
    ) -> Union[List[int], str]:
        """
        Run algorithm muliple time to find converging point.

        Arguments:
            initial_population (list): Population of chromosomes.
            mutation_probabilty (float): Chance for each gene that mutation takes place while mutation.
            fitness_func (function): Python function which takes chromosome and calculates probablistic fitness.
            max_iter (int): Exit condition to break infinite loop condition in case of non-convergence.

        Returns:
            (str): Best solution.
        """
        self.path_generated = 0
        self.__fitness_func = fitness_func
        self.__population_fitness_cache = []
        self.mean_history = []
        self.max_history = []
        self.inverse_max_history = []

        population = initial_population
        population_fitness = self.__population_fitness(population)
        max_fitness = max(self.__population_fitness_cache)
        max_fitness_child = None

        if self.__crossover_type == "Binary":
            k = 2
        else:
            k = 1

        for _ in range(max_iter):
            parents = self.select(population, population_fitness, n_parents=k)
            child = self.crossover(parents)
            child = self.mutate(child, mutation_probabilty)

            child_fitness = self.__fitness_func(child)
            population.append(child)
            population_fitness.append(child_fitness)

            self.mean_history.append(mean(self.__population_fitness_cache))
            self.max_history.append(max_fitness)
            self.inverse_max_history.append(1 / max_fitness)

            if child_fitness > max_fitness:
                max_fitness = child_fitness
                max_fitness_child = child

        return max_fitness_child

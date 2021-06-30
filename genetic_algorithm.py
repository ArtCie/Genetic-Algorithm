import numpy as np


class GeneticAlg:
    """Genetic Algorithm class - defines all"""
    def __init__(self, variable_number, population_size, dx, cross_probability, mutation_probability, start_point, finish_point, max_iter):
        self.variable_number = variable_number
        self.population_size = population_size
        self.dx = float(dx)
        self.cross_probability = cross_probability
        self.mutation_probability = mutation_probability
        self.start_point = start_point
        self.finish_point = finish_point
        self.max_iter = max_iter

    @staticmethod
    def obj_func(x, fun):
        """Method returns value of given function at (x1, x2)"""
        x1, x2 = x
        return eval(fun)

    def nbits(self):
        """Method returns number of bits needed to create individual"""
        length = (self.finish_point - self.start_point) / self.dx + 1
        b_ = int(np.ceil(length/self.dx)).bit_length()
        dx_new = (self.finish_point - self.start_point) / (2 ** b_ - 1)
        return b_, dx_new

    def gen_population(self):
        """Method generates random population"""
        return np.random.randint(2, size=(self.population_size, self.variable_number * self.finish_point))

    def decode_individual(self, individual):
        """Method decodes individual from binary"""
        result = np.array_split(individual, self.variable_number)
        for i, j in enumerate(result):
            new = np.array2string(np.array(result[i]), separator="")[1:-1]
            new = new.replace("\n ", "")
            result[i] = self.start_point + (int(new, 2) * self.dx)
        return result

    def evaluate_population(self, func, pop, eval_fun):
        """Method returns all values from individuals - to evaluate them"""
        return np.apply_along_axis(func, -1,
                                   np.apply_along_axis(self.decode_individual, 1, pop), eval_fun)

    @staticmethod
    def get_best(pop, evaluated_pop):
        """Method returns best individual"""
        best_value = np.min(evaluated_pop)
        best_index = np.where(evaluated_pop == best_value)
        best_individual = pop[best_index[0]]
        return *best_individual, best_value

    @staticmethod
    def roulette(pop, evaluated_pop):
        """Roulette method - get random individuals with given probability - the better individual, the better score"""
        evaluated_pop = evaluated_pop * (-1)
        if evaluated_pop.min() < 0:
            evaluated_pop += np.abs(evaluated_pop.min()) + 1
        probabilities = np.array(evaluated_pop / np.sum(evaluated_pop))
        return np.array(pop[np.random.choice(pop.shape[0], len(pop), replace=True, p=probabilities), :])

    @staticmethod
    def cross(pop, pk):
        """Cross method - with given probability cross two individuals at random place"""
        for i in range(int(len(pop) / 2)):
            if np.random.choice([0, 1], p=[1 - pk, pk]):
                cut_position = np.random.randint(1, len(pop[0]))
                first = np.append(pop[2 * i][:cut_position], pop[2 * i + 1][cut_position:])
                second = np.append(pop[2 * i + 1][:cut_position], pop[2 * i][cut_position:])
                pop[2 * i] = first
                pop[2 * i + 1] = second
        return pop

    @staticmethod
    def mutate(pop, pm):
        """Mutate method - using XOR operator and given probability change random bit"""
        prob = np.random.choice([1, 0], size=(len(pop), len(pop[0])), p=[pm, 1 - pm])
        return np.logical_xor(pop, prob).astype(int)

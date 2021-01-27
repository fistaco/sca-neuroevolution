import numpy as np
import random as rand

from data_processing import sample_data
from nn_genome import NeuralNetworkGenome


class GeneticAlgorithm:
    def __init__(self, max_gens, pop_size, mut_power, mut_rate, crossover_rate,
                 mut_power_decay_rate, truncation_proportion, atk_set_size):
        self.max_gens = max_gens
        self.pop_size = pop_size
        self.mut_power = mut_power
        self.mut_rate = mut_rate
        self.crossover_rate = crossover_rate
        self.mut_power_decay_rate = mut_power_decay_rate
        self.truncation_proportion = truncation_proportion
        self.atk_set_size = atk_set_size

        # Maintain the population and all offspring in self.population
        # The offspring occupy the second half of the array
        self.population = np.empty(pop_size*2, dtype=object)

        # Store useful information
        self.best_fitness_per_gen = np.empty(max_gens, dtype=np.int16)

    def run(self, nn, x_atk_full, y_atk_full, ptexts, true_subkey, subkey_i=2):
        """
        Runs the genetic algorithm with the parameters it was constructed with
        and returns the best found individual.
        """
        self.initialise_population(nn)

        # Track generational information
        gen = 0
        best_fitness = 256
        best_individual = None

        while gen < self.max_gens and best_fitness > 0:
            # Randomly sample the attack set
            print("Sampling data....")
            (x_atk, y_atk) = \
                sample_data(self.atk_set_size, x_atk_full, y_atk_full)

            # Fitness evaluation and updating of best individual
            print("Evaluating fitness values...")
            self.evaluate_fitness(x_atk, y_atk, ptexts, true_subkey, subkey_i)
            fitnesses = np.vectorize(lambda x: x.fitness)(self.population)
            best_idx = fitnesses.argmin()
            best_fitness = fitnesses[best_idx]
            best_individual = self.population[best_idx]

            print("Selecting individuals...")
            # Rest of GA main loop, i.e. selection & offspring production
            self.population[:self.pop_size] = self.roulette_wheel_selection()  # TODO: add truncation selection?
            print("Producing offspring...")
            self.population[self.pop_size:] = self.produce_offpsring()

            # Track useful information
            # TODO: Test best individual on a separate test to check generalisation
            self.best_fitness_per_gen[gen] = best_fitness
            print(f"Best fitness in generation {gen}: {best_fitness}")
            gen += 1
        
        return best_individual

    def initialise_population(self, nn):
        """
        Initialises a population of NNs with the given architecture parameters.
        """
        for i in range(len(self.population)):
            self.population[i] = NeuralNetworkGenome(nn)
            # TODO: maybe initialise weights randomly?

    def evaluate_fitness(self, x_atk, y_atk, ptexts, true_subkey, subkey_idx):
        """
        Computes and sets the current fitness value of each individual in the
        population and list of offspring.
        """
        # TODO: Paralellise
        for indiv in self.population:
            indiv.evaluate_fitness(x_atk, y_atk, ptexts, true_subkey)

    def roulette_wheel_selection(self):
        """
        Selects potentially strong individuals by assigning each of them a
        selection probability that scales with their fitness. This is also
        known as fitness proportionate selection.

        This method assumes the fitness values have already been computed in
        the current generation.
        """
        new_population = np.empty(self.pop_size, dtype=object)

        # Compute the sum of all individuals' fitness values
        fitness_sum = np.sum(np.fromiter(
            (indiv.fitness for indiv in self.population),
            dtype=np.int16
        ))

        # Compute individual fitness probabilities and sum them up cumulatively
        # to construct sections of the roulette wheel, while keeping in mind
        # that a smaller fitness value is better in our scenario.
        size = self.pop_size*2
        cumulative_select_probs = np.zeros(size, dtype=float)
        cumulative_prob = 0.0
        for i in range(size):
            # Max. key rank is 255, so invert fitness f as (255 - f)
            inverted_fitness = 255 - self.population[i].fitness
            cumulative_prob += inverted_fitness / fitness_sum

            cumulative_select_probs[i] = cumulative_prob

        print(f"======== Cumulative probs: ========\n{cumulative_select_probs}")

        # Build the new population by "spinning the wheel" repeatedly
        for i in range(self.pop_size):
            idx = cumulative_select_probs.searchsorted(np.random.uniform())
            new_population[i] = self.population[idx]

        return new_population

    def tournament_selection(self, t_size=3):
        """
        Selects potentially strong individuals by performing fitness-based
        tournaments with replacement.
        """
        new_population = np.empty(self.pop_size, dtype=object)
        for i in range(self.pop_size):
            # Pick a random participant, which starts as the winner by default
            winner = np.random.choice(self.population)

            # Compare (t_size - 1) more individuals based on their fitness 
            for j in range(t_size - 1):
                indiv = np.random.choice(self.population)
                if indiv.fitness > winner.fitness:
                    winner = indiv
            
            new_population[i] = winner
        
        return new_population
    
    def produce_offpsring(self):
        """
        Produces and returns offspring by applying either mutation or crossover
        on each individual, doubling the total population size.
        """
        offspring = np.empty(self.pop_size, dtype=object)
        for i in range(self.pop_size):
            parent0 = self.population[i]

            if np.random.uniform() < 0.5:
                parent1 = np.random.choice(self.population[:self.pop_size])
                offspring[i] = parent0.crossover(parent1)
            else:
                offspring[i] = parent0.mutate(self.mut_power, self.mut_rate)
        
        return offspring

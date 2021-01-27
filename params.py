# Genetic algorithm parameters
MUTATION_POWER = 0.03
MUTATION_RATE = 0.04
CROSSOVER_RATE = 0.5
MAX_GENERATIONS = 100
POPULATION_SIZE = 1000
TRUNCATION_PROPORTION = 0.4
TOURNAMENT_SIZE = 3

# Params specific to the approach from the paper by Morse and Stanley (2016)
MUTATION_POWER_DECAY = 0.99
FITNESS_INHERITANCE_DECAY = 0.2

# SCA-specific parameters
ATTACK_SET_SIZE = 1000  # Size of the set used for fitness evaluation of an NN

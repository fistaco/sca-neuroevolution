from metrics import MetricType


# Genetic algorithm parameters
MUTATION_POWER = 0.03
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.5
MAX_GENERATIONS = 50
POPULATION_SIZE = 182
TRUNCATION_PROPORTION = 1.0
SELECTION_FUNCTION = "tournament"
TOURNAMENT_SIZE = 3

# Params specific to the approach from the paper by Morse and Stanley (2016)
MUTATION_POWER_DECAY = 0.999
FITNESS_INHERITANCE_DECAY = 0.2
APPLY_FITNESS_INHERITANCE = False
BALANCED_TRACES = False

# SCA-specific parameters
ATTACK_SET_SIZE = 8000  # Size of the set used for fitness evaluation of an NN
METRIC_TYPE = MetricType.INCREMENTAL_KEYRANK

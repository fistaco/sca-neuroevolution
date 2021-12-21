from enum import Enum

import numpy as np


class CrossoverType(Enum):
    """
    The type of crossover used in the NASCTY-CNNS genetic algorithm.
    """
    ONEPOINT = 0
    PARAMETERWISE = 1


class PoolType(Enum):
    """
    The type of pooling layer used for some neural network.
    """
    AVERAGE = 0
    MAX = 1

    def mutate_with_prob(self, mut_prob):
        """
        Returns AVERAGE if the current pool type is MAX, and returns MAX if the
        current pool type is AVERAGE.
        """
        return PoolType(not self.value) if np.random.uniform() < mut_prob \
            else self

    @staticmethod
    def random():
        """
        Returns a random pooling layer type.
        """
        return PoolType(np.random.randint(2))
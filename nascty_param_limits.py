class NasctyParamLimits:
    """
    A singleton object encapsulating the minimum and maximum value for each
    numerical parameter that can be modified during evolution with the NASCTY
    genetic algorithm.
    """
    def __init__(self):
        self.n_conv_blocks_min = 0
        self.n_conv_blocks_max = 5
        self.n_dense_layers_min = 1
        self.n_dense_layers_max = 5

        self.n_filters_min = 2
        self.n_filters_max = 128
        self.filter_size_min = 1
        self.filter_size_max = 100
        self.pool_size_min = 2
        self.pool_size_max = 100
        self.pool_stride_min = 2
        self.pool_stride_max = 100

        self.n_dense_neurons_min = 1
        self.n_dense_neurons_max = 30

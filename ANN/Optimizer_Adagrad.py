import numpy as np

class Optimizer_Adagrad:
    #initialize otpimizer with learning rate, decay, and epsilon
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    #call once before parameter updates
    def pre_update_params(self):
        #if we use learning rate decay over time
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    #update parameters
    def update_params(self, layer):
        #If layer does not have cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            #create cache for biases
            layer.bias_cache = np.zeros_like(layer.biases)

        #update caches for weights and biases
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        
        #vanilla SGD with square rooted cache and epsilon
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
    

    #call after parameter updates
    def post_update_params(self):
        self.iterations += 1

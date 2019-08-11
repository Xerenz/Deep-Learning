import random
import numpy as np

def sigmoid(z):
    # z is Numpy array
    return 1.0 / 1.0 + np.exp(-z)

def sigmoid_prime(z):
    return sigmoid(z)*(1 - sigmoid(z))


class NeuralNetwork(object):

    def __init__(self, sizes):
        self.sizes = sizes
        self.layers = len(sizes)
        self.weights = [np.random.randn(x, y) 
            for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(x, 1) 
            for x in sizes[1:]]

    def feedforward(self, a):
        '''Given a is the input to the network layer
        return a corresponding output.'''
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)

        return a

    def SGD(self, trainig_data, epoch, eta, minibatch_size, test_data=None):
        '''Train neural network using the trainig data
        a tuple (x, y) containig the trainig examples.
        eta is the learning rate. If test data is provided
        use it to track progress at each epoch.''' 

        n = len(trainig_data)

        for i in range(epoch):
            random.shuffle(trainig_data)

            mini_batches = [trainig_data[k:k + minibatch_size]
                           for k in range(0, n, minibatch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                n_test = len(test_data)

                print('Epoch {0} : {1} / {2}'.format(
                    i, self.evaluate(test_data), n_test
                ))
            else:
                print('Epoch {} complete'.format(i))

    def update_mini_batch(self, mini_batch, eta):
        '''Make the slight required changes to the 
        weights and biases of the network for each
        "mini_batch". Apply backpropogation to each 
        trainig example in the "mini_batch".'''

        # initialise step chages
        step_weights = [np.zeros(w.shape) for w in self.weights]
        step_biases = [np.zeros(b.shape) for b in self.biases]

        for x, y in mini_batch:
            d_weights, d_biases = self.backpropogation(x, y)

            step_weights = [sw + dw for sw, dw in zip(step_weights, d_weights)]
            step_biases = [sb + db for sb, db in zip(step_biases, d_biases)]

        # update current weights and biases
        self.weights = [w - (eta/len(mini_batch))*sw for 
                        w, sw in zip(self.weights, step_weights)]
        self.biases = [b - (eta/len(mini_batch))*sb for
                        b, sb in zip(self.biases, step_biases)]



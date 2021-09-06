from scipy import *
from scipy.linalg import norm, pinv
#import random
from matplotlib import pyplot as plt
import math
import numpy as np
from keras.layers import Layer
from keras import backend as K


class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
#         print(input_shape)
#         print(self.units)
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


class RBF:

    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [np.random.uniform (-1, 1, indim) for i in range(numCenters)]
        self.beta = 10
        self.W = np.random.random((self.numCenters, self.outdim))

    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return math.exp(-self.beta * norm(c-d)**2)

    def _calcAct(self, X):
        # calculate activations of RBFs
        G = np.zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G

    def train(self, X, Y):
        """ X: matrix of dimensions n x indim
            y: column vector of dimension n x 1 """

        # choose random center vectors from training set
        rnd_idx = np.random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i,:] for i in rnd_idx]

        #print("center", self.centers)
        # calculate activations of RBFs
        G = self._calcAct(X)
        #print(G)

        # calculate output weights (pseudoinverse)
        self.W = np.dot(pinv(G), Y)

    def test(self, X):
        """ X: matrix of dimensions n x indim """

        G = self._calcAct(X)
        Y = np.dot(G, self.W)
        return Y


if __name__ == '__main__':
    # ----- 1D Example ------------------------------------------------
    '''
    n = 100

    x = np.mgrid[-1:1:complex(0,n)].reshape(n, 1)
    y = []
    for i in range(len(x)):
        #y[i] = math.sin(3*(x[i]+0.5)**3 - 1)
        y.append([math.sin(3*(x[i]+0.5)**3 - 1)])
        #y = np.concatenate((y, math.sin(3*(x[i]+0.5)**3 - 1)), axis=0)
    y = np.array(y)
    # set y and add random noise
    #y = math.sin(3*(x+0.5)**3 - 1)
    # y += random.normal(0, 0.1, y.shape)

    # rbf regression
    rbf = RBF(1, 10, 1)
    rbf.train(x, y)
    z = rbf.test(x)

    # plot original data
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'k-')

    # plot learned model
    plt.plot(x, z, 'r-', linewidth=2)

    # plot rbfs
    plt.plot(rbf.centers, np.zeros(rbf.numCenters), 'gs')

    for c in rbf.centers:
        # RF prediction lines
        cx = np.arange(c-0.7, c+0.7, 0.01)
        cy = [rbf._basisfunc(np.array([cx_]), np.array([c])) for cx_ in cx]
        plt.plot(cx, cy, '-', color='gray', linewidth=0.2)

    plt.xlim(-1.2, 1.2)
    plt.show()
    '''

import numpy as np
import scipy as sp

class Maha:

    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon

    def fit(self, train):
        self.train = train
        cov = np.cov(self.train.T)
        cov[np.isnan(cov)] = 0
        if self.epsilon:
            N = self.train.shape[1]
            eps_mtrx = np.zeros((N, N), float)
            np.fill_diagonal(eps_mtrx, self.epsilon)
            self.inv_cov = sp.linalg.inv(cov + eps_mtrx)
        else:
            self.inv_cov = sp.linalg.inv(cov)

    def predict(self, test):
        x_minus_mu = test - np.mean(self.train, axis=0)
        left_term = np.dot(x_minus_mu, self.inv_cov)
        mahal = np.dot(left_term, x_minus_mu.T)
        return mahal

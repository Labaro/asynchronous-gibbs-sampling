import numpy as np
from scipy.stats import multivariate_normal
from time import time
from .problem import Problem
from copy import deepcopy


class ToyProblem(Problem):
    def __init__(self, tau=1, phi=0.5, alpha=1):
        self.dim = 8
        super().__init__("Toy Problem", self.dim)
        self.oldx = 10 * np.ones(self.dim) + 0.01 * np.arange(1, self.dim + 1)
        self.tau = tau
        self.phi = phi
        self.alpha = alpha
        self.sigma = np.array(
            [[tau ** 2 * np.exp(-phi * abs(i - j) ** alpha) for i in range(self.dim)] for j in range(self.dim)])
        self.mu = np.zeros(self.dim)
        self.assigned_dims = np.arange(self.dim)

    def pdf(self, x):
        # print(multivariate_normal.pdf(x, self.mu, self.sigma))
        return multivariate_normal.pdf(x, self.mu, self.sigma)

    def cpdf(self, x, y):
        # print(multivariate_normal.pdf(x + y, 2 * self.mu, self.sigma))
        return multivariate_normal.pdf(x + y, 2 * self.mu, self.sigma)

    def split(self, nb_worker):
        subproblems = []
        n = self.dim // nb_worker
        for i in range(nb_worker):
            dims = self.assigned_dims[i * n: (i + 1) * n]
            copy = deepcopy(self)
            copy.assigned_dims = dims
            subproblems.append(copy)
        return subproblems

    def sample(self):
        t = time()
        i = np.random.choice(self.assigned_dims)

        sidx = np.array([j == i for j in range(self.dim)])
        midx = np.array([j != i for j in range(self.dim)])

        mu1 = self.mu[sidx]
        mu2 = self.mu[midx]

        x2 = self.oldx[midx]

        sigma11 = self.sigma[:, sidx][sidx]
        sigma22 = self.sigma[:, midx][midx]
        sigma12 = self.sigma[:, midx][sidx]
        sigma21 = self.sigma[:, sidx][midx]

        sigma22inv = np.linalg.inv(sigma22)

        mustar = mu1 + sigma12.dot(sigma22inv).dot(x2 - mu2)
        sigmastar = sigma11 - sigma12.dot(sigma22inv).dot(sigma21)

        self.oldx[sidx] = np.random.normal(mustar[0], sigmastar[0, 0])

        return deepcopy(self.oldx), time() - t

    def send(self):
        return self.oldx

    def receive(self, message, assigned_dims):
        x_i, x_s = message
        x_p = deepcopy(self.oldx)
        x_p[assigned_dims] = x_i[assigned_dims]
        # print(self.pdf(x_p) * self.cpdf(self.oldx, x_s) / (self.pdf(self.oldx) * self.cpdf(x_p, x_s)))
        p = min(1, self.pdf(x_p) * self.cpdf(self.oldx, x_s) / (self.pdf(self.oldx) * self.cpdf(x_p, x_s)))
        # print(p)
        if np.random.random() <= p:
            self.oldx[assigned_dims] = x_i[assigned_dims]
            # self.oldx = deepcopy(message)
            return True, p
        else:
            return False, p

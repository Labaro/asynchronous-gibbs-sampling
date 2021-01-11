import numpy as np


class Worker:
    def __init__(self, id, problem, sending_iter):
        self.id = id
        self.problem = problem
        self.history = [problem.oldx.copy()]
        self.k = 0
        self.time = 0
        self.sending_iter = sending_iter
        self.received = True
        self.approx = np.mean(self.history)

    def __eq__(self, other):
        return self.id == other.id

    def sample(self):
        value, t = self.problem.sample()
        self.history.append(value)
        self.time += t
        self.k += 1

    def receive(self, message, i):
        self.received = True
        return self.problem.receive(message, i)

    def send(self):
        #if self.sending_iter[self.k - 1]:
        if self.received:
            self.received = False
            return self.problem.send(), self.history[self.k - 1]
        return None

    def stop_criteria(self):
        self.approx = np.mean(self.history)
        return np.linalg.norm(self.approx) <= 1 / 4

    def mixing_rate(self):
        history = np.array(self.history)
        nMC = history.shape[0]
        n = 0 # int(0.2 * nMC)
        history = history[n:]
        cum_norm = np.linalg.norm(
            np.cumsum(history, axis=0) / np.repeat(np.arange(n + 1, nMC + 1), self.problem.dim).reshape(nMC - n,
                                                                                                        self.problem.dim),
            axis=1, ord=1) / self.problem.dim
        # print(np.flip(cum_norm)[:100])
        mask = cum_norm <= 1 / 4
        # print(np.argmin(np.flip(mask)), np.argmax(np.flip(mask)), np.all(mask))
        return nMC - np.argmin(np.flip(mask))

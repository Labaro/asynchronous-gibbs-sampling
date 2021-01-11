import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from .worker import Worker


class Control:
    def __init__(self, nb_workers, problem, max_delay, send_proba, nMC, cyclic=False):
        self.nb_workers = nb_workers
        self.message_buffer = {i: [] for i in range(nb_workers)}
        self.workers = []
        self.nMC = nMC
        self.problem = problem
        assert problem.dim % nb_workers == 0, f"The problem is of dimension {problem.dim} which cannot be divided equally between {nb_workers}"
        split_problem = problem.split(nb_workers)
        for i in range(nb_workers):
            self.workers.append(Worker(i, split_problem[i], bernoulli.rvs(send_proba, size=self.nMC).tolist()))
        self.delays = np.random.randint(0, max_delay, size=nb_workers)
        self.message_counter = 0
        self.messages_received = 0
        self.p = []
        self.cyclic = cyclic

    def run(self):
        for k in range(self.nMC):
            if k % 1000 == 0:
                print("Iteration", k)
            for i, w in enumerate(self.workers):
                message_received = "None"
                for iteration, message, assigned_dims in self.message_buffer[i]:
                    if w.k >= iteration:
                        message_received = f"Worker {w.id} received an update for parameter {assigned_dims}"
                        self.message_buffer[i].pop(0)
                        received, p = w.receive(message, assigned_dims)
                        self.messages_received += int(received)
                        self.p.append(p)
                w.sample()
                message = w.send()
                message_send = "None"
                if message is not None:
                    message_send = f"Worker {w.id} send a message"
                    self.update_message(message, w, w.problem.assigned_dims)
                # print(w.id, message_received, message_send)

    def update_message(self, message, sender, assigned_dims):
        for i, w in enumerate(self.workers):
            if self.cyclic:
                if (i - sender.id) % self.nb_workers == 1: # Cyclic workload
                    if self.nb_workers > 2:
                        random_dim = np.sort(
                            np.random.choice([j for j in range(self.problem.dim) if j not in w.problem.assigned_dims and j not in sender.problem.assigned_dims], 1,
                                             replace=False))
                    else:
                        random_dim = np.empty(0)
                    delay = np.random.randint(self.delays[i] + 1)
                    self.message_buffer[i].append((w.k + delay, message, np.sort(np.concatenate([assigned_dims, random_dim]))))
                    self.message_buffer[i].sort(key=lambda x: x[0])
                    self.message_counter += 1
            else:
                if w != sender: # Random workload
                    delay = np.random.randint(self.delays[i] + 1)
                    self.message_buffer[i].append((w.k + delay, message, assigned_dims))
                    self.message_buffer[i].sort(key=lambda x: x[0])
                    self.message_counter += 1

    def show_results(self):
        max_time = 0
        mixing_rate = []
        for i, w in enumerate(self.workers):
            # print(np.mean(np.array(w.history), axis=0))
            if w.time > max_time:
                max_time = w.time
            #print(np.mean(w.history[int(0.5 * self.nMC):], axis=0))
            rate = w.mixing_rate()
            mixing_rate.append(rate)
            #print(rate)
            history = np.array(w.history)
            # print(np.repeat(np.arange(1, self.nMC + 2), 8).reshape(self.nMC + 1, 8).shape)
            # print(history)
            print(np.mean(history[int(0.2 * self.nMC):], axis=0))

            x = np.arange(1, self.nMC + 2)
            y = np.linalg.norm(
                np.cumsum(history, axis=0) / np.repeat(np.arange(1, self.nMC + 2), 8).reshape(self.nMC + 1, 8),
                axis=1, ord=1) / 8
            plt.plot(x,y)
            plt.axhline(0, linestyle='--', color='k')
            plt.xscale('log')
            plt.xlabel('Iterations')
            plt.yscale('log')
            plt.ylabel("$\ell 1$-norm")
            plt.title(f"$\ell 1$-norm between the average coefficient of worker {w.id} and the true vector.")
            plt.show()

            plt.plot(history[:, w.problem.assigned_dims])
            plt.axhline(0, linestyle='--', color='k')
            plt.xlabel("Iterations")
            plt.ylabel("Value of the variables")
            plt.title(f"History of the variables sampled by worker {w.id}")
            plt.show()
        plt.hist(self.p, bins=25, density=True)
        plt.title("Metropolis-Hasting acceptance probability histogram")
        plt.show()
        print("Total time for sampling", max_time)
        print("Total messages send:", self.message_counter)
        print("Total messages accepted:", self.messages_received)
        return mixing_rate, max_time, self.message_counter, self.messages_received


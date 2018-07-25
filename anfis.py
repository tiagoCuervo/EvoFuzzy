import numpy as np
import matplotlib.pyplot as plt


class ANFIS:
    def __init__(self, n_inpts, n_rules):
        self.n = n = n_inpts
        self.m = m = n_rules
        self.mus = np.zeros(shape=(1, n * m))
        self.sigmas = np.zeros(shape=(1, n * m))
        self.y = np.zeros(shape=(1, m))

    def setmfs(self, means, stdevs, sequents):
        self.mus = means
        self.sigmas = stdevs
        self.y = sequents

    def rule_firing(self, x):
        # Evaluates membership functions on each input for the whole batch
        F = np.reshape(np.exp(-0.5 * ((np.tile(x, (1, self.m)) - self.mus) ** 2) / (self.sigmas ** 2)),
                       (-1, self.m, self.n))
        # Gets the firing strenght of each rule by applying T-norm (product in this case)
        return np.prod(F, axis=2)

    def defuzzify(self, w):
        return np.sum(self.y * w, axis=1) / np.clip(np.sum(w, axis=1), a_min=1e-12, a_max=1e12)

    def infer(self, x):
        return self.defuzzify(self.rule_firing(x))

    def plotmfs(self):
        mus = np.reshape(self.mus, (self.m, self.n))
        sigmas = np.reshape(self.sigmas, (self.m, self.n))
        xn = np.linspace(np.min(mus) - 3 * np.max(sigmas), np.max(mus) + 3 * np.max(sigmas), 1000)
        for r in range(self.m):
            if r % 4 == 0:
                plt.figure(figsize=(11, 6), dpi=80)
            plt.subplot(2, 2, (r % 4) + 1)
            ax = plt.subplot(2, 2, (r % 4) + 1)
            ax.set_title("Rule %d, sequent center: %f" % ((r + 1), self.y[r]))
            for i in range(self.n):
                plt.plot(xn, np.exp(-0.5 * ((xn - mus[r, i]) ** 2) / (sigmas[r, i] ** 2)))

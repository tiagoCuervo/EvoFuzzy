import time
import matplotlib.pyplot as plt
from diffevo import differential_evolution
from anfis import ANFIS
from fobj import *


# Mackey-Glass series computation
def mackey(n_iters):
    x = np.zeros((n_iters,))
    x[0:30] = 0.23 * np.ones((30,))
    t_s = 30
    for i in range(30, n_iters - 1):
        a = x[i]
        b = x[i - t_s]
        y = ((0.2 * b) / (1 + b ** 10)) + 0.9 * a
        x[i + 1] = y
    return x


# Generate dataset
D = 4  # number of regressors
T = 1  # delay
N = 2000  # Number of points to generate
mg_series = mackey(N)[499:]  # Use last 1500 points
data = np.zeros((N - 500 - T - (D - 1) * T, D))
lbls = np.zeros((N - 500 - T - (D - 1) * T,))

for t in range((D - 1) * T, N - 500 - T):
    data[t - (D - 1) * T, :] = [mg_series[t - 3 * T], mg_series[t - 2 * T], mg_series[t - T], mg_series[t]]
    lbls[t - (D - 1) * T] = mg_series[t + T]

# Creates the inference system
m = 16  # number of rules
fis = ANFIS(D, m)
n_params = 2 * (m * D) + m  # Total number of parameters (genome size)


# Evaluates the objective function
def eval_objective(params):
    # From the parameter vector (genome) gets each set of parameters (means, standard deviations and sequent singletons)
    mus = params[0:fis.m * fis.n]
    sigmas = params[fis.m * fis.n:2 * fis.m * fis.n]
    y = params[2 * fis.m * fis.n:]
    # Sets the FIS parameters to the ones on the genome
    fis.setmfs(mus, sigmas, y)
    pred = fis.infer(data)
    loss = 1 - nse(pred, lbls)
    return loss


# Runs the evolution cycle
start_time = time.time()
result = list(differential_evolution(eval_objective, bounds=[(-2, 2)] * n_params, gens=10))
end_time = time.time()
print('Evolution time: %f' % (end_time - start_time))
# Gets the last genome
best_params = result[-1][0]
best_mus = best_params[0:fis.m * fis.n]
best_sigmas = best_params[fis.m * fis.n:2 * fis.m * fis.n]
best_y = best_params[2 * fis.m * fis.n:2 * fis.m * fis.n + fis.m]
# Sets the FIS parameters to the ones of the last genome
fis.setmfs(best_mus, best_sigmas, best_y)
# Predicts output for the training set
best_pred = fis.infer(data)
# Plots the real and predicted one series
plt.plot(mg_series)
plt.plot(best_pred)
plt.legend(['Real', 'Predicted'])
plt.show()
print('Best fitness: %f' % result[-1][1])

import numpy as np


def differential_evolution(fobj, bounds, mut=0.8, crossprob=0.7, popsize=30, gens=1000, mode='best/1'):
    # Gets number of parameters (length of genome vector)
    num_params = len(bounds)
    # Initializes the population genomes with values drawn from uniform distribution in the range [0,1]
    pop = np.random.rand(popsize, num_params)
    # Gets the boundaries for each parameter to scale the population genomes
    min_b, max_b = np.asarray(bounds).T
    # Scales the population genomes from the range [0,1] to the range specified by the parameter boundaries
    diff = np.fabs(min_b - max_b)
    pop_scaled = min_b + pop * diff
    # Evaluates fitness for each individual in the population by calculating the objective to minimize
    unfitness = np.asarray([fobj(ind) for ind in pop_scaled])
    # Gets the best individual of the population
    best_idx = np.argmin(unfitness)
    best = pop_scaled[best_idx]
    for i in range(gens):
        print('Best unfitness in generation %d: %f' % (i + 1, unfitness[best_idx]))
        # For each individual:
        for j in range(popsize):
            # Selects three individuals from the population different than himself(no jerking off) for reproduction
            if mode == 'best/1':
                idxs = [idx for idx in range(popsize) if (idx != j and idx != best_idx)]
                a = best
                b, c = pop[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(a + mut * (b - c), 0, 1)
            elif mode == 'best/2':
                idxs = [idx for idx in range(popsize) if (idx != j and idx != best_idx)]
                a = best
                b, c, d, e = pop[np.random.choice(idxs, 4, replace=False)]
                # Generates a mutant by applying the differential mutation (and clips to keep in range [0,1])
                mutant = np.clip(a + mut * (b - c + d - e), 0, 1)
            elif mode == 'rand/1':
                idxs = [idx for idx in range(popsize) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                # Generates a mutant by applying the differential mutation (and clips to keep in range [0,1])
                mutant = np.clip(a + mut * (b - c), 0, 1)
            elif mode == 'rand/2':
                idxs = [idx for idx in range(popsize) if idx != j]
                a, b, c, d, e = pop[np.random.choice(idxs, 5, replace=False)]
                # Generates a mutant by applying the differential mutation (and clips to keep in range [0,1])
                mutant = np.clip(a + mut * (b - c + d - e), 0, 1)
            # Selects parameters of the individual to crossover with the mutant with the probability of crossover
            cross_points = np.random.rand(num_params) < crossprob
            # If some parameter results to need crossover ...
            if not np.any(cross_points):
                # selects the index of that parameter for crossover
                cross_points[np.random.randint(0, num_params)] = True
            # The parameters of the individual's genome that require crossover gets changed for those of the mutant,
            # producing a new individual
            trial = np.where(cross_points, mutant, pop[j])
            # Scales the genome of the new individual from the range [0,1] to the range specified by the parameter
            # boundaries
            trial_denorm = min_b + trial * diff
            # Evaluates fitness of new individual
            f = fobj(trial_denorm)
            # If better than the previous one, keeps the new one
            if f < unfitness[j]:
                unfitness[j] = f
                pop[j] = trial
                # If better than the best one so far, updates the record
                if f < unfitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, unfitness[best_idx]

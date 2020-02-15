import itertools
import random
import numpy as np

if __name__ == '__main__':

    from bandits import LinUCB
    from run_bandits import *

    profiles, click_rates = load_ctr_data('ctr_data.txt')
    num_arms = click_rates[0].shape[0]
    num_articles = len(click_rates)
    context_dim = profiles[0].shape[0]

    random.seed(0)
    np.random.seed(0)

    # Define hyperparameters grid
    grid = {'alpha': np.arange(0.0, 2.1, 0.1).tolist()}
    # generate combinations
    params = sorted(grid)
    combinations = itertools.product(*(grid['alpha'],))

    print("Evaluating LIN-UCB strategy on CTR data.")

    optim_run = 0.
    optim_rsum = 0.
    optim_params = dict()
    for run, (alpha,) in enumerate(combinations, 1):
        bandit = LinUCB(num_arms, context_dim, alpha)
        print("Run {}: alpha={}".format(run, alpha))
        pulled_arms, rewards = \
            train(bandit, click_rates, profiles, contextual=True)

        rsum = np.sum(rewards)
        if rsum > optim_rsum:
            print('New best')
            optim_rsum = rsum
            optim_run = run
            optim_params['alpha'] = alpha

    print("Done. Best run: {}, best hyperparameters: alpha={}"
          .format(optim_run, optim_params['alpha']))


# Grid scale 1 best: alpha = 0.2: cumulated reward = 1427
# Grid scale 2 best: alpha = 0.16: cumulated reward = 1432

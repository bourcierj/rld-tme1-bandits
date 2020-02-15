import itertools
import random
import numpy as np

if __name__ == '__main__':

    from bandits import EpsilonGreedy
    from run_bandits import *

    profiles, click_rates = load_ctr_data('ctr_data.txt')
    num_arms = click_rates[0].shape[0]
    num_articles = len(click_rates)
    context_dim = profiles[0].shape[0]

    random.seed(0)
    np.random.seed(0)

    # Define hyperparameters grid
    grid = {'epsilon': [0.3, 0.2, 0.15, 0.10, 0.05, 0.01],
            'epsilon_decay': [1., 0.9999, 0.999, 0.99, 0.9]
            }
    # generate combinations
    params = sorted(grid)
    combinations = itertools.product(*(grid['epsilon'], grid['epsilon_decay']))

    print("Evaluating EPSILON-GREEDY strategy on CTR data.")

    optim_run = 0.
    optim_rsum = 0.
    optim_params = dict()
    for run, (epsilon, epsilon_decay) in enumerate(combinations, 1):
        bandit = EpsilonGreedy(num_arms, epsilon, epsilon_decay)
        print("Run {}: epsilon={}, epsilon_decay={}".format(run, epsilon, epsilon_decay))
        pulled_arms, rewards = \
            train(bandit, click_rates, profiles, contextual=False)

        rsum = np.sum(rewards)
        if rsum > optim_rsum:
            print('New best')
            optim_rsum = rsum
            optim_run = run
            optim_params['epsilon'] = epsilon
            optim_params['epsilon_decay'] = epsilon_decay

    print("Done. Best run: {}, best hyperparameters: epsilon={}, epsilon_decay={}"
          .format(optim_run, optim_params['epsilon'], optim_params['epsilon_decay']))

# Grid scale 1 best: epsilon=0.15, epsilon_decay=0.99: cumulated reward = 1334

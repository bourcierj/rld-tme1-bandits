import itertools
import random
from tqdm import tqdm

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

    N_TRIALS = 30
    n_configs = len(grid['epsilon'])*len(grid['epsilon_decay'])
    print("Tuning EPSILON-GREEDY strategy on CTR data.")
    print('Number of trials per config: {}'.format(N_TRIALS))
    print('Number of configs: {}'.format(n_configs))
    optim_run = 0.
    optim_metric = -float('inf')
    optim_params = dict()

    for run, (epsilon, epsilon_decay) in enumerate(combinations, 1):
        metric = 0.
        pbar = tqdm(range(N_TRIALS), desc='Config {}: epsilon={}, epsilon_decay={}'
                    .format(run, epsilon, epsilon_decay))
        # print("Run {}: epsilon={}, epsilon_decay={}".format(run, epsilon, epsilon_decay))
        for trial in pbar:
            bandit = EpsilonGreedy(num_arms, epsilon, epsilon_decay)
            pulled_arms, rewards = \
                train(bandit, click_rates, profiles, contextual=False, verbose=False)

            #Get average reward at the end
            ravg = np.sum(rewards) / len(rewards)
            metric += ravg
            pbar.set_postfix({'ravg': ravg})

        metric /= N_TRIALS
        print('Config {}: Mean average reward: {}'.format(run, metric))
        # Select on the metric averaged over trials
        if metric > optim_metric:
            print('Config {}: New best'.format(run))
            optim_metric = metric
            optim_run = run
            optim_params['epsilon'] = epsilon
            optim_params['epsilon_decay'] = epsilon_decay

    print("Done. Best config: {}, average reward: {}, hyperparams: {}"
          .format(optim_run, optim_metric, optim_params))

# Best config: 13, average reward: 0.2562834417068091,
# hyperparams: {'epsilon': 0.15, 'epsilon_decay': 0.999}

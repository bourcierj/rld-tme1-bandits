import itertools
import random
from tqdm import tqdm

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
    grid = {'alpha': np.arange(0.11, 0.26, 0.01).tolist()}

    # generate combinations
    params = sorted(grid)
    combinations = itertools.product(*(grid['alpha'],))

    n_configs = len(grid['alpha'])
    print("Tuning LIN-UCB strategy on CTR data.")
    print('Number of configs: {}'.format(n_configs))
    optim_run = 0.
    optim_metric = -float('inf')
    optim_params = dict()

    for run, (alpha,) in enumerate(combinations, 1):
        metric = 0.
        print("Config {}: alpha={}".format(run, alpha))
        bandit = LinUCB(num_arms, context_dim, alpha)
        pulled_arms, rewards = \
            train(bandit, click_rates, profiles, contextual=True, verbose=False)

        #Get average reward at the end
        ravg = np.sum(rewards) / len(rewards)
        metric = ravg
        print('Config {}: Mean average reward: {}'.format(run, metric))
        # Select on the metric averaged over trials
        if metric > optim_metric:
            print('Config {}: New best'.format(run))
            optim_metric = metric
            optim_run = run
            optim_params['alpha'] = alpha

    print("Done. Best config: {}, average reward: {}, hyperparams: {}"
          .format(optim_run, optim_metric, optim_params))

# Grid scale 1: Best config: 3, average reward: 0.2854156877066617,
# hyperparams: {'alpha': 0.2}
# Grid scale 2: Best config: 6, average reward: 0.286418224719455,
# hyperparams: {'alpha': 0.16}

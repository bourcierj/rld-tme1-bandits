import argparse
import random
import numpy as np

from bandits import *
from run_bandits import *

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Bandits algorithm on a click-through "
                                     "rate dataset.")
    parser.add_argument('--plot', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    profiles, click_rates = load_ctr_data('ctr_data.txt')
    num_arms = click_rates[0].shape[0]
    num_articles = len(click_rates)
    context_dim = profiles[0].shape[0]

    random.seed(0)
    np.random.seed(0)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn')
        fig, axs = plt.subplots(2, figsize=(12,8))

    print("Expected reward of every advertiser:")

    print(("{:<12}" + "{:<12d}" * num_arms).format("Advertiser", *range(num_arms)))
    print(("{:<12}" + "{:<12.5f}" * num_arms).format("Reward", *get_expected_rewards(click_rates)))

    for algorithm in ('random', 'static-best', 'optimal', 'epsilon-greedy', 'ucb',
                      'ucb-v', 'lin-ucb'):

        kwargs = dict()
        if algorithm == 'epsilon-greedy':
            kwargs = dict(epsilon=0.1, epsilon_decay=0.999)
        if algorithm == 'lin-ucb':
            kwargs = dict(alpha=0.16)

        bandit = get_bandit_instance(algorithm, num_arms, context_dim, **kwargs)

        baseline = algorithm in {'random', 'static-best', 'optimal'}
        contextual = algorithm in {'lin-ucb'}

        print()
        print("Evaluating {} strategy on CTR data.".format(algorithm.upper()))

        pulled_arms, rewards = \
            train(bandit, click_rates, profiles, contextual)

        if args.plot:
            # Get average rewards
            avg_rewards = get_moving_average(rewards)
            # Plot
            ls = None
            if baseline:
                ls = ':'
            elif contextual:
                ls = '--'
            axs[0].plot(range(num_articles), avg_rewards, linestyle=ls, label=algorithm)
            axs[1].scatter(range(num_articles), pulled_arms, s=3., alpha=0.8, facecolor=None)

    if args.plot:
        axs[0].set_ylabel('Avg reward')
        axs[0].legend(loc='lower right')
        axs[1].set_ylabel('Pulled arm')
        axs[1].set_yticks(range(10))
        plt.tight_layout()
        plt.show()

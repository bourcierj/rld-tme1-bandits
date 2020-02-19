import argparse
import random
import numpy as np

from bandits import *
from run_bandits import *

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Bandits algorithms on a click-through "
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
        fig1, ax1 = plt.subplots(figsize=(12,8))
        fig2, ax2 = plt.subplots(3, 2, figsize=(12, 8))

    print("Expected reward of every advertiser:")

    print(("{:<12}" + "{:<12d}" * num_arms).format("Advertiser", *range(num_arms)))
    print(("{:<12}" + "{:<12.5f}" * num_arms).format("Reward", *get_expected_rewards(click_rates)))
    ax_i = 0
    for algorithm in ('random', 'static-best', 'optimal', 'epsilon-greedy', 'ucb',
                      'ucb-v', 'lin-ucb'):

        # Get any hyperparameters
        kwargs = dict()
        if algorithm == 'epsilon-greedy':
            kwargs = dict(epsilon=0.15, epsilon_decay=0.999)
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
            avg_rewards = moving_avg(rewards)
            # Get mean confidence intervals
            conf = .99
            mci_rewards = moving_mean_confidence_interval(rewards, conf)

            # Plot
            ls = None
            if baseline:
                ls = ':'
            line, = ax1.plot(range(num_articles), avg_rewards, linestyle=ls, label=algorithm)
            ax1.fill_between(range(num_articles), mci_rewards[0], mci_rewards[1], alpha=0.4)

            if algorithm != 'random':
                ax2i = ax2.flat[ax_i]
                ax2i.scatter(range(num_articles), pulled_arms, s=3., color=line.get_color(),
                             alpha=0.8, facecolor=None)
                ax2i.set_title(algorithm)
                ax2i.set_ylabel('Pulled arm')
                ax2i.set_yticks(range(num_arms))
                ax_i += 1

    if args.plot:
        ax1.set_ylabel('Avg reward (w/ {:.0%} confidence interval)'.format(conf))
        ax1.set_ylim((0.020798076458847778, 0.3216468127248515))
        ax1.legend(loc=(0.81, 0.22))
        fig1.savefig('figures/avg_rewards_all_mci.png', dpi=300, bbox_inches='tight')
        plt.show()
        fig2.tight_layout()
        fig2.savefig('figures/pulled_arms_all.png', dpi=300, bbox_inches='tight')
        plt.show()

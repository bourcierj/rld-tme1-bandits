import argparse
import random
import numpy as np

from bandits import *

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Bandits algorithm on a click-through "
                                     "rate dataset.")
    parser.add_argument('--algorithm', '-a', type=str, default='epsilon-greedy')
    parser.add_argument('--plot', action='store_true')
    return parser.parse_args()


def load_ctr_data(filename):
    """Loads the data file of click-through rates."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [l.split(':') for l in lines]
    profiles = [np.array(l[1].split(';'), dtype=np.float) for l in lines]
    click_rates = [np.array(l[2].rstrip().split(';'), dtype=np.float)
                   for l in lines]

    return (profiles, click_rates)

def get_expected_rewards(click_rates):
    """Returns the expected rewards of every arm from the click rates data
    """
    return np.mean(np.stack(click_rates, 0), 0)


def train(bandit, click_rates, profiles, contextual=True):

    if not contextual:
        profiles = [None]* len(click_rates)

    rsum = 0.  # cumulated reward
    # to compute the regret:
    best_arm = np.argmax(get_expected_rewards(click_rates))
    best_rsum = 0.  # cumulated reward of the optimal arm
    pulled_arms = list()
    rewards_list = list()

    for j, (rewards, contexts) in enumerate((zip(click_rates, profiles)), 1):

        if hasattr(bandit, 'cheater'):
            # pull an arm knowing the rewards at current iteration
            arm = bandit.choose(rewards)
        else:
            # pull an arm
            if contextual:
                arm = bandit.choose(contexts)
            else:
                arm = bandit.choose()

        reward = rewards[arm]
        # Update the agent internal state
        bandit.update(arm, reward)

        rsum += reward
        best_rsum += rewards[best_arm]
        pulled_arms.append(arm)
        rewards_list.append(reward)


    print("Cumulated reward: {}".format(rsum))
    print("Empirical regret: {}".format(best_rsum - rsum))

    # Post-process results
    # ravg_list = (rsum_list / np.arange(1, j+1)).tolist()
    # regret_list = np.subtract(best_rsum_list, rsum_list).tolist()

    return pulled_arms, rewards_list
    # rsum_list, ravg_list, regret_list

def get_bandit_instance(name, num_arms, **kwargs):

    mapping = {'random': lambda: Random(num_arms),
               'static-best': lambda: StaticBest(num_arms),
               'optimal': lambda: Optimal(num_arms),
               'epsilon-greedy': lambda: EpsilonGreedy(num_arms, **kwargs),
               'ucb': lambda: UCB(num_arms),
               'ucb-v': lambda: UCBV(num_arms),
               'lin-ucb': lambda: LinUCB(num_arms)}

    try:
        return mapping[name]()
    except KeyError:
        raise ValueError("Unknown algorithm: {}".format(name))


if __name__ == '__main__':

    args = parse_args()
    profiles, click_rates = load_ctr_data('ctr_data.txt')
    num_arms = 10
    num_articles = 5000

    random.seed(0)
    np.random.seed(0)

    kwargs = dict()
    if args.algorithm == 'epsilon-greedy':
        kwargs = dict(epsilon=0.1, epsilon_decay=0.999)
    bandit = get_bandit_instance(args.algorithm, num_arms, **kwargs)
    contextual = False
    if args.algorithm == 'lin-ucb':
        contextual = True

    print("Evaluating {} strategy on CTR data.".format(args.algorithm.upper()))

    pulled_arms, rewards = \
        train(bandit, click_rates, profiles, contextual)

    if args.plot:
        import matplotlib.pyplot as plt
        # import seaborn as sns
        # sns.set()
        plt.style.use('seaborn')
        # Get average rewards
        avg_rewards = np.cumsum(rewards) / np.arange(1, num_articles+1)
        # Plot
        fig, axs = plt.subplots(2, figsize=(12,8))
        axs[0].plot(range(num_articles), avg_rewards, label=args.algorithm)
        # plt.plot(range(num_articles), regret_list, label='Regret')
        axs[0].set_ylabel('Avg reward')
        axs[0].legend()
        axs[1].scatter(range(num_articles), pulled_arms, s=3., alpha=0.8, facecolor=None, label=args.algorithm)
        axs[1].set_ylabel('Pulled arm')
        axs[1].set_yticks(range(10))
        plt.tight_layout()
        plt.show()

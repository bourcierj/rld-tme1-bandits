import argparse
import random
import numpy as np

from bandits import *

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Bandits algorithm on a click-through "
                                     "rate dataset.")
    parser.add_argument('--algorithm', '-a', type=str, default='epsilon-greedy')
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
    best_rsum = 0.
    pulled_arms = list()
    rsum_list = list()
    best_rsum_list = list()
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
        # update regret
        best_rsum += rewards[best_arm]
        pulled_arms.append(arm)
        rsum_list.append(rsum)
        best_rsum_list.append(best_rsum)

    print("Cumulated reward: {}".format(rsum))
    print("Empirical regret: {}".format(best_rsum - rsum))

    # Post-process results
    ravg_list = (np.cumsum(rsum_list) / np.arange(1, j+1)).tolist()
    regret_list = np.subtract(best_rsum_list, rsum_list).tolist()

    return pulled_arms, rsum_list, ravg_list, regret_list


if __name__ == '__main__':

    args = parse_args()
    profiles, click_rates = load_ctr_data('ctr_data.txt')
    num_arms = 10

    random.seed(0)
    np.random.seed(0)

    contextual = False
    # Baselines
    if args.algorithm == 'random':
        bandit = Random(num_arms)
    elif args.algorithm == 'static-best':
        bandit = StaticBest(num_arms)
    elif args.algorithm == 'optimal':
        bandit = Optimal(num_arms)
    # Bandits strategies
    elif args.algorithm == 'epsilon-greedy':
        args.epsilon = 0.1
        args.epsilon_decay = 1.
        bandit = EpsilonGreedy(num_arms, args.epsilon, args.epsilon_decay)
    elif args.algorithm == 'ucb':
        bandit = UCB(num_arms)
    elif args.algorithm == 'ucb-v':
        bandit = UCBV(num_arms)
    elif args.algorithm == 'lin-ucb':
        bandit = LinUCB(num_arms)
        contextual = True
    else:
        raise ValueError("Unknown algorithm: {}".format(args.algorithm))

    print("Evaluating {} strategy on CTR data".format(args.algorithm.upper()))

    pulled_arms, rsum_list, ravg_list, regret_list = \
        train(bandit, click_rates, profiles, contextual)



    # print("Evaluating bandit {} on CTR data".format(bandit.__class__.__name__))

    # rsum = 0.  # cumulated reward
    # # to compute the regret:
    # best_arm = np.argmax(get_expected_rewards(click_rates))
    # best_rsum = 0.
    # pulled_arms = list()
    # from tqdm import tqdm
    # for rewards, contexts in tqdm(zip(click_rates, profiles)):

    #     if hasattr(bandit, 'cheater'):
    #         # pull an arm knowing the rewards at current iteration
    #         arm = bandit.choose(rewards)
    #     else:
    #         # pull an arm
    #         if contexts is not None:
    #             arm = bandit.choose(contexts)
    #         else:
    #             arm = bandit.choose()

    #     reward = rewards[arm]
    #     # Update the agent internal state
    #     bandit.update(arm, reward)
    #     rsum += reward
    #     # update regret
    #     best_rsum += rewards[best_arm]
    #     pulled_arms.append(arm)

    # print("Cumulated reward: {}".format(rsum))
    # print("Empirical regret: {}".format(best_rsum - rsum))

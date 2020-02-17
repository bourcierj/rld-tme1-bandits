import random
import numpy as np

class BaseBandit():
    """Base class for multi-armed bandits agent"""
    def __init__(self, num_arms):
        self.num_arms = num_arms

    def choose(self):
        """Choose an arm
        Returns:
            (int): the arm chosen
        """
        raise NotImplementedError()

    def update(self, arm, reward):
        """Updates the internal state of the agent.
        Args:
            arm (int): the arm chosen
            reward (float): the reward perceived
        """
        raise NotImplementedError()


class BaseContextualBandit(BaseBandit):
    """Base class for contextual bandits agents
    """
    def __init__(self, num_arms):
        super(BaseContextualBandit, self).__init__(num_arms)

    def choose(self, context):
        """Choose an arm
        Args:
            context (numpy.array): context for current iteration
        Returns:
            (int): the arm chosen
        """
        raise NotImplementedError()


class Random(BaseBandit):
    """Simple random agent"""
    def __init__(self, num_arms):
        super(Random, self).__init__(num_arms)

    def choose(self):
        return np.random.randint(self.num_arms)

    def update(self, arm, reward):
        pass


class StaticBest(BaseBandit):
    """At every iteration, choose the arm with best cumulated click rates.
       This is a cheating agent who sees all rewards at every time step.
    """
    def __init__(self, num_arms):
        super(StaticBest, self).__init__(num_arms)
        # means of click rates
        self.means = np.zeros(num_arms)
        # number of time each arm has been chosen
        self.clicks = np.zeros(num_arms)
        # current iteration
        self.t = 0
        # inform that this agent in a cheater
        self.cheater = True

    def choose(self, rewards):
        # greedy choice of the arm
        arm = np.argmax(self.means)
        self.last_rewards = rewards
        return arm

    def update(self, arm, reward):
        self.clicks += 1
        self.t += 1
        self.means = self.means + 1/self.t * (self.last_rewards - self.means)


class Optimal(BaseBandit):
    """At every iteration, choose the arm with best click rate at this
       iteration. This is a cheating agent who seens all rewards at every
       time step.
    """
    def __init__(self, num_arms):
        super(Optimal, self).__init__(num_arms)
        # inform that this agent in a cheater
        self.cheater = True

    def choose(self, rewards):
        return np.argmax(rewards)

    def update(self, arm, reward):
        pass


class EpsilonGreedy(BaseBandit):
    """Epsilon-greedy strategy.
       With proba epsilon, explore, and with proba (1 - epsilon), exploit.
    """
    def __init__(self, num_arms, epsilon=0.0, epsilon_decay=1.):
        super(EpsilonGreedy, self).__init__(num_arms)
        # means of click rates
        self.means = np.zeros(num_arms)
        # number of time each arm has been chosen
        self.clicks = np.zeros(num_arms)
        # current iteration
        self.t = 0
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def choose(self):
        # init phase: try every arm once
        if self.t < self.num_arms:
            return self.t
        if np.random.rand() < self.epsilon:  # explore
            arm = np.random.randint(self.num_arms)
        else:  # exploit
            arm = np.argmax(self.means)

        self.epsilon *= self.epsilon_decay
        return arm

    def update(self, arm, reward):
        self.t += 1
        self.clicks[arm] += 1
        self.means[arm] = self.means[arm] + 1/self.clicks[arm] \
                          *(reward - self.means[arm])


class UCB(BaseBandit):
    """UCB (Upper confidence bound) strategy.
    """
    def __init__(self, num_arms):
        super(UCB, self).__init__(num_arms)
        # means of click rates
        self.means = np.zeros(num_arms)
        # number of time each arm has been chosen
        self.clicks = np.zeros(num_arms)
        # current iteration
        self.t = 0

    def choose(self):
        # init phase: try every arm once
        if self.t < self.num_arms:
            return self.t
        # return arm with maximum upper confidence bound
        t = self.t+1
        bounds = self.means + (2* np.log(t) / self.clicks)** 0.5

        arm = np.argmax(bounds)
        return arm

    def update(self, arm, reward):
        self.t += 1
        self.clicks[arm] += 1
        self.means[arm] = self.means[arm] + 1/self.clicks[arm] \
            *(reward - self.means[arm])


class UCBV(BaseBandit):
    """UCB-V (Upper confidence bound on variance) strategy.
    """
    def __init__(self, num_arms):
        super(UCBV, self).__init__(num_arms)
        # means of rewards
        self.means = np.zeros(num_arms)
        # means of squared rewards
        self.means_sq = np.zeros(num_arms)
        # number of time each arm has been chosen
        self.clicks = np.zeros(num_arms)
        # current iteration
        self.t = 0

    def choose(self):
        # init phase: try every arm once
        if self.t < self.num_arms:
            return self.t
        # compute variances
        variances = self.means_sq - self.means**2
        t = self.t+1
        # return arm with maximum upper confidence bound
        bounds = self.means + (2* np.log(t) * variances / self.clicks)** 0.5 \
            + np.log(t) / (2* self.clicks)

        arm = np.argmax(bounds)
        return arm

    def update(self, arm, reward):
        self.t += 1
        self.clicks[arm] += 1
        self.means[arm] = self.means[arm] + 1/self.clicks[arm] \
            *(reward - self.means[arm])
        self.means_sq[arm] = self.means_sq[arm] + 1/self.clicks[arm] \
            *(reward**2 - self.means_sq[arm])


class LinUCB(BaseContextualBandit):
    """Lin-UCB contextual strategy. From Li et al. 2010 (https://arxiv.org/abs/1003.0146)
    """
    #@todo
    def __init__(self, num_arms, context_dim, alpha=0.5):
        super(LinUCB, self).__init__(num_arms)

        self.context_dim = context_dim
        self.alpha = alpha
        # Initialize the context matrix A and reward vector b
        self.A = np.tile(np.eye(context_dim, context_dim), (num_arms, 1, 1))
        self.A_inv = self.A  # inverse of A
        self.b = np.zeros((num_arms, context_dim))
        # weight parameters
        self.theta = np.zeros((num_arms, context_dim))
        # current iteration
        self.t = 0

    def choose(self, context):
        """Note: context here is a global vector: same context for every possible arms."""
        # # init phase: try every arm once
        # if self.t < self.num_arms:
        #     return self.t

        # Compute upper bounds on the linear function output
        bounds = np.array([
            self.theta[a].dot(context) + self.alpha * \
                (context.dot(self.A_inv[a]).dot(context)) ** 0.5
            for a in range(self.num_arms) ])

        assert(bounds.shape == (self.num_arms,))

        self.lcontext = context

        arm = np.argmax(bounds)
        return arm

    def update(self, arm, reward):

        context = self.lcontext
        # Update matrices A and vectors b
        self.A[arm] += context.reshape(-1, 1).dot(context.reshape(1, -1))
        self.b[arm] += reward * context
        self.A_inv = np.linalg.inv(self.A)
        # Update parameters
        self.theta = np.stack([
            self.A_inv[a].dot(self.b[a]) for a in range(self.num_arms)], axis=0
        )
        assert(self.theta.shape == (self.num_arms, self.context_dim))

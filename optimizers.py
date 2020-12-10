from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.stats as stats
import torch
from config.utils import torch_truncated_normal
import collections


class Optimizer:
    def __init__(self, *args, **kwargs):
        pass

    def setup(self, cost_function):
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def obtain_solution(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclass.")


class CEMOptimizer(Optimizer):

    def __init__(self, sol_dim, max_iters, popsize, num_elites, cost_function,
                 upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.sol_dim, self.max_iters, self.popsize, self.num_elites = sol_dim, max_iters, popsize, num_elites

        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha

        self.cost_function = cost_function

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

    def reset(self):
        pass

    def obtain_solution(self, init_mean, init_var):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        mean, var, t = init_mean, init_var, 0
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(var))
        while (t < self.max_iters) and np.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            samples = X.rvs(size=[self.popsize, self.sol_dim]) * np.sqrt(constrained_var) + mean
            samples = samples.astype(np.float32)
            costs = self.cost_function(samples)

            elites = samples[np.argsort(costs)][:self.num_elites]

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            t += 1
        return mean


class DiscreteRandomOptimizer(Optimizer):

    def __init__(self, sol_dim, max_iters, popsize, num_elites, cost_function,
                 upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.sol_dim, self.max_iters, self.popsize, self.num_elites = sol_dim, max_iters, popsize, num_elites

        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha

        self.cost_function = cost_function

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

    def reset(self):
        pass

    def obtain_solution(self, init_mean, possible_actions):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean: The starting mean, this is unused.
            possible_actions (np.ndarray): The possible actions this discrete env allows
        """
        samples = np.random.choice(np.arange(possible_actions.shape[-1]),
                                   size=[self.popsize * (self.sol_dim // possible_actions.shape[-1])],
                                   replace=True)
        samples = possible_actions[samples]
        samples = samples.astype(np.float32).reshape(self.popsize, self.sol_dim)
        costs = self.cost_function(samples)
        elite = samples[np.argsort(costs)][:1]
        return elite.flatten()


class DiscreteCEMOptimizer(Optimizer):

    def __init__(self, sol_dim, max_iters, popsize, num_elites, cost_function,
                 upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.sol_dim, self.max_iters, self.popsize, self.num_elites = sol_dim, max_iters, popsize, num_elites

        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha

        self.cost_function = cost_function

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

        self.one_hot = None
        self.discrete_actions = None

    def reset(self):
        pass

    # This function should return concrete action (e.g. [-1, 0])
    # instead of action id, because the action will be fed into 
    # _predict_next_obs
    def sample_from_categorical(self, probs, possible_actions, num_samples):
        print('====')
        print(probs)
        probs += 0.001 # prevent torch from thinking anything is < 0
        samples = torch.distributions.categorical.Categorical(probs=torch.from_numpy(probs)).sample([num_samples])
        print(samples)
        print(samples.shape)
        print(possible_actions)

        
        samples = possible_actions[samples.numpy()].astype(np.float32)
        print(samples)
        print(samples.shape)
        return samples

    # performing one hot encoding
    def obtain_solution(self, init_mean, possible_actions):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            possible_actions (np.ndarray): The possible actions this discrete env allows
        """
        mean, t = init_mean, 0
        print(mean)
        print(mean.shape)
        assert mean.shape == (25, 5)
        while t < self.max_iters:
            print('t: ' + str(t))

            # Should this return a series of concrete actions
            samples = self.sample_from_categorical(mean, possible_actions, self.popsize)
            # Shape here should be (400, 25, 2)
            # samples = samples.reshape(self.popsize, self.sol_dim)
            samples = samples.reshape(self.popsize, mean.shape[0], int(self.sol_dim / mean.shape[0]))
            print('popsize: ' + str(self.popsize))
            print('sample shape: ' + str(samples.shape))

            # Is the shape here correct?
            costs = self.cost_function(samples)
            print('costs:')
            print(costs)
            print(costs.shape)
            print('mean:')
            print(mean)
            print(*mean.shape)
            print(self.num_elites)
            elites = samples[np.argsort(costs)][:self.num_elites]
            print('elites')
            print(elites)
            print(elites.shape)
            print(self.sol_dim)
            print(self.popsize)

            # FIXME: encounter dimensionality error here.
            # FIXME: need 1 more step between elites and mean
            # elites are concrete action samples. [0, -1], not [0.2, 0.2, 0.2, 0.2, 0.2]
            print(elites.shape)
            elites_prob = np.zeros((self.num_elites, *mean.shape))
            for i in range(elites.shape[0]): # num_elites
                for j in range(elites.shape[1]): # plan_hor
                    if np.array_equal(elites[i, j], np.array([0,0])):
                        elites_prob[i, j, 0] = 1.0
                    elif np.array_equal(elites[i, j], np.array([0,-1])):
                        elites_prob[i, j, 1] = 1.0
                    elif np.array_equal(elites[i, j], np.array([0,1])):
                        elites_prob[i, j, 2] = 1.0
                    elif np.array_equal(elites[i, j], np.array([-1,0])):
                        elites_prob[i, j, 3] = 1.0
                    elif np.array_equal(elites[i, j], np.array([1,0])):
                        elites_prob[i, j, 4] = 1.0
            print('elites_prob')
            print(elites_prob)
            # It is trying to do 2000 -> (40, 25, 5)
            # elites = elites.reshape(self.num_elites, *mean.shape)
            # But it should do 2000 -> (40, 25, 2)
            # elites = elites.reshape(self.num_elites, mean.shape[0], int(self.sol_dim / mean.shape[0]))


            new_mean = np.mean(elites_prob, axis=0)
            print('new_mean')
            print(type(new_mean))
            print(new_mean.shape)
            print(new_mean)

            # Update mean with a constant alpha term
            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            print('self.alpha')
            print(type(self.alpha))
            print(self.alpha)

            print('mean')
            print(type(mean))
            print(mean.shape)
            print(mean)

            normalization_factor = mean.sum(axis=-1)
            print('normalization factor')
            print(normalization_factor)
            mean = mean / normalization_factor[:, np.newaxis]
            print('final mean')
            print(mean)
            t += 1

        return mean

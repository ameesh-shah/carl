from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import trange

import numpy as np
import torch

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from MPC import MPC
from env.pointmass import PointmassEnv

'''
Explore with MPC + intrinsic motivation mixed reward
'''
class ExploreMPC(MPC):

    def __init__(self, params):
        super().__init__(params)

        # Override cost function to be exploration reward
        self.optimizer.cost_function = self._compile_cost

    @torch.no_grad()
    def _compile_cost(self, ac_seqs):
        # ac_seqs shape: (popsize, plan_hor, ac_dim)
        # Preprocess
        ac_seqs = torch.from_numpy(ac_seqs).float().to(TORCH_DEVICE)

        # Reshape ac_seqs so that it's amenable to parallel compute
        # ac_seqs final shape: (self.plan_hor, npart * pop_size, ac_dim) = (25, 8000, 1 [cartpole])
        ac_seqs = ac_seqs.view(-1, self.plan_hor, self.dU)
        transposed = ac_seqs.transpose(0, 1)
        expanded = transposed[:, :, None]
        tiled = expanded.expand(-1, -1, self.npart, -1)
        ac_seqs = tiled.contiguous().view(self.plan_hor, -1, self.dU)

        # Expand current observation
        # self.sy_cur_obs: (obs_dim,)
        # cur_obs final shape: (npart * pop_size, obs_dim) = (8000, 1 [cartpole])
        cur_obs = torch.from_numpy(self.sy_cur_obs).float().to(TORCH_DEVICE)

        print('Current observation: ' + str(cur_obs))
        print('Action sequence: ' + str(ac_seqs))

        cur_obs = cur_obs[None]
        cur_obs = cur_obs.expand(self.optimizer.popsize * self.npart, -1)

        intrinsic_cost = self._compile_cost_intrinsic(ac_seqs, cur_obs)
        supervised_cost = self._compile_cost_reward(ac_seqs, cur_obs)

        print('Intrinsic cost: ' + str(intrinsic_cost))
        print('Supervised cost: ' + str(supervised_cost))

        # TODO: make weight on each a parameter
        return (intrinsic_cost + supervised_cost) / 2.0

    @torch.no_grad()
    def _compile_cost_intrinsic(self, ac_seqs, cur_obs):
        raise NotImplementedError("Intrinsic motivation function not implemented!")

    @torch.no_grad()
    def _compile_cost_reward(self, ac_seqs, cur_obs):
        """Computes supervised reward (environment reward (baseline) or unsafe reward (ours)).

        Args:
            ac_seqs (torch.Tensor):
            cur_obs (torch.Tensor):
        Returns:
            cost (ndarray):

        """
        costs = torch.zeros(self.optimizer.popsize, self.npart, device=TORCH_DEVICE)

        for t in range(self.plan_hor):
            cur_acs = ac_seqs[t]

            # next_obs shape: (npart * pop_size, obs_shape) = (8000, 4)
            next_obs = self._predict_next_obs(cur_obs, cur_acs)
            # cost shape: (npart * pop_size, obs_shape)

            # Reward calculation but do special treatment for PointmassEnv
            if (isinstance(self.env, PointmassEnv)):
                # print(next_obs)
                _, reward = self.env.get_dist_and_reward(next_obs[..., :2])
                # print(reward)
                # print(reward.shape)
                # The cost should mostly be ones, if the reward is sparse.
                cost = -1 * np.sum(reward) + self.ac_cost_fn(cur_acs)
                cost = torch.Tensor(cost)
            else:
                cost = self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs)

            if self.mode == 'test' and not self.no_catastrophe_pred:  # use catastrophe prediction during adaptation
                # catastrophe_cost_fn masks `cost`
                #   if there is a catastrophe, the cost for that timestep is increased by COLLISION_COST=1e4 (configured in config/{env}.py)
                #   else, the cost is `cost`
                # self.percentile (default 100 during train / 50 during test) controls when we mark a state as trajectory
                #   e.g. if 50, we get marked as catastrophe if (predicted) next_obs catastrophe >= .5
                #   setting it lower results in more risk-averse planning (we avoid states if there is even a small prob of catastrophe)
                cost = self.catastrophe_cost_fn(next_obs, cost, self.percentile)
            elif self.mode == 'train' and self.unsafe_pretraining:
                catastrophe_prob = next_obs[..., -1]
                cost = -(100 * catastrophe_prob)  # negate so cost is in [-100, 0] (lowest cost for catastrophe_prob=1)
            # cost: (popsize, npart)
            cost = cost.view(-1, self.npart)
            costs += cost
            cur_obs = self.obs_postproc2(next_obs)

        # Replace nan with high cost
        costs[costs != costs] = 1e6
        mean_cost = costs.mean(dim=1)
        return mean_cost.detach().cpu().numpy()
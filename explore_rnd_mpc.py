from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import trange

import torch
import numpy as np

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from explore_base_mpc import ExploreMPC
from rnd_model import RNDModel
from env.pointmass import PointmassEnv


class ExploreRNDMPC(ExploreMPC):

    def __init__(self, params):
        super().__init__(params)

        # Set up exploration model
        self.exploration_enabled = True
        self.exploration_model = RNDModel(ob_dim=self.dO)
        self.gamma = 1e-2

    @torch.no_grad()
    def _compile_cost_intrinsic(self, ac_seqs, cur_obs):
        nopt = ac_seqs.shape[0]
        ac_seqs = torch.from_numpy(ac_seqs).float().to(TORCH_DEVICE)

        # Reshape ac_seqs so that it's amenable to parallel compute
        ac_seqs = ac_seqs.view(-1, self.plan_hor, self.dU)
        transposed = ac_seqs.transpose(0, 1)
        expanded = transposed[:, :, None]
        tiled = expanded.expand(-1, -1, self.npart, -1)
        ac_seqs = tiled.contiguous().view(self.plan_hor, -1, self.dU)

        # Expand current observation
        #cur_obs = torch.from_numpy(self.sy_cur_obs).float().to(TORCH_DEVICE)
        cur_obs = cur_obs[None]
        cur_obs = cur_obs.expand(nopt * self.npart, -1)
        costs = torch.zeros(nopt, self.npart, device=TORCH_DEVICE)
        for t in range(self.plan_hor):
            cur_acs = ac_seqs[t]
            next_obs = self._predict_next_obs(cur_obs, cur_acs)
            cost = self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs)
            # FIXME: add exploreation bonus
            # FIXME: minimize cost to maximize reward
            if (self.exploration_enabled) :
                print('======= Cost Before =======')
                print(type(cost))
                print(cost)

                expl_bonus = self.gamma * self._get_expl_bonus(next_obs)

                print('======= Expl Bonus  =======')
                print(type(expl_bonus))
                print(expl_bonus)

                cost = cost - expl_bonus

                print('======= Cost After  =======')
                print(type(cost))
                print(cost)
            if self.mode == 'test' and not self.no_catastrophe_pred: #use catastrophe prediction during adaptation
                cost = self.catastrophe_cost_fn(next_obs, cost, self.percentile)
            cost = cost.view(-1, self.npart)
            costs += cost
            cur_obs = self.obs_postproc2(next_obs)
        # replace nan with high cost
        costs[costs != costs] = 1e6
        if self.no_catastrophe_pred:
            # Discounted reward sum calculation for CARL (Reward). At self.percentile == 100, this is normal PETS
            if self.percentile <= 100:
                k_percentile = -(-costs).kthvalue(k=max(int((self.percentile/100) * costs.shape[1]), 1), dim=1)[0]
                cost_mask = costs <  k_percentile.view(-1, 1).repeat(1, costs.shape[1])
            else:
                k_percentile = costs.kthvalue(k=max(int(((200 - self.percentile)/100) * costs.shape[1]), 1), dim=1)[0]
                cost_mask = costs >  k_percentile.view(-1, 1).repeat(1, costs.shape[1])
            costs[cost_mask] = 0
            discounted_sum = costs.sum(dim=1)
            costs[cost_mask] = float('nan')
            lengths = torch.sum(~torch.isnan(costs), dim=1).float()
            mean_cost = discounted_sum / lengths
        else:
            mean_cost = costs.mean(dim=1)
        return mean_cost.detach().cpu().numpy()

    def _get_expl_bonus(self, next_obs):
        """
        print('***********')
        print('next_obs type: ' + str(type(next_obs)))
        print(next_obs)
        """
        expl_bonus = self.exploration_model(next_obs).detach()
        self.exploration_model.update(next_obs)
        print('Exploration bonus: ' + str(expl_bonus))
        return expl_bonus

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import trange

import torch

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from MPC import MPC
from rnd_model import RNDModel


class ExploreRNDMPC(MPC):

    def __init__(self, params):
        super().__init__(params)

        # Set up exploration model
        self.exploration_enabled = True
        self.exploration_model = RNDModel(ob_dim=self.dO)
        self.gamma = 1e-2

    @torch.no_grad()
    def _compile_cost_intrinsic(self, ac_seqs):
        nopt = ac_seqs.shape[0]
        ac_seqs = torch.from_numpy(ac_seqs).float().to(TORCH_DEVICE)

        # Reshape ac_seqs so that it's amenable to parallel compute
        ac_seqs = ac_seqs.view(-1, self.plan_hor, self.dU)
        transposed = ac_seqs.transpose(0, 1)
        expanded = transposed[:, :, None]
        tiled = expanded.expand(-1, -1, self.npart, -1)
        ac_seqs = tiled.contiguous().view(self.plan_hor, -1, self.dU)

        # Expand current observation
        cur_obs = torch.from_numpy(self.sy_cur_obs).float().to(TORCH_DEVICE)
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

    @torch.no_grad()
    def _compile_cost_reward(self, ac_seqs):
        nopt = ac_seqs.shape[0]
        ac_seqs = torch.from_numpy(ac_seqs).float().to(TORCH_DEVICE)

        # Reshape ac_seqs so that it's amenable to parallel compute
        ac_seqs = ac_seqs.view(-1, self.plan_hor, self.dU)
        transposed = ac_seqs.transpose(0, 1)
        expanded = transposed[:, :, None]
        tiled = expanded.expand(-1, -1, self.npart, -1)
        ac_seqs = tiled.contiguous().view(self.plan_hor, -1, self.dU)

        # Expand current observation
        cur_obs = torch.from_numpy(self.sy_cur_obs).float().to(TORCH_DEVICE)
        cur_obs = cur_obs[None]
        cur_obs = cur_obs.expand(nopt * self.npart, -1)
        costs = torch.zeros(nopt, self.npart, device=TORCH_DEVICE)

        obs_vars = []
        obs_means = []

        for t in range(self.plan_hor):
            cur_acs = ac_seqs[t]

            # next_obs shape: (npart * pop_size, obs_shape) = (8000, 4)
            # mean, var shape: (num_nets, npart * popsize / num_nets, obs_shape) = (5, 1600, 4)
            # calculate variance over all bootstraps
            next_obs, (mean, var) = self._predict_next_obs(cur_obs, cur_acs, return_mean_var=True)
            # each of `popsize` CEM samples is a different action, so we shouldn't avg states over popsize
            # mean: (num_nets, npart / num_nets, popsize, obs_shape)
            mean = mean.view(self.model.num_nets, self.npart // self.model.num_nets,
                    nopt, self.dO)
            # obs_mean_per_bootstrap: (num_nets, popsize, obs_shape)
            # average next state prediction (over all particles) for each bootstrap
            obs_mean_per_bootstrap = torch.mean(mean, dim=1)

            obs_means.append(obs_mean_per_bootstrap)
            obs_vars.append(var)

            cur_obs = self.obs_postproc2(next_obs)

        # Calculate max aleatoric var for each state component to standardize
        obs_vars = torch.stack(obs_vars)
        assert obs_vars.shape == (self.plan_hor, self.model.num_nets,
            (self.npart * nopt) // self.model.num_nets, self.dO)
        # obs_vars: (self.plan_hor, num_net, npart * popsize / num_nets, obs_shape) -> (-1, obs_shape)
        obs_vars = obs_vars.view(-1, self.dO)
        # w_base: (obs_shape,)
        w_base, _ = torch.max(obs_vars, dim=0)

        # Calculate variance over bootstrap mean predictions
        # obs_means: (self.plan_hor, num_nets, popsize, obs_shape)
        obs_means = torch.stack(obs_means)
        assert obs_means.shape == (self.plan_hor, self.model.num_nets, nopt, self.dO)
        # Disagreement (var) across bootstraps about the next state indicates epistemic uncertainty
        # obs_epistemic_var: (self.plan_hor, popsize, obs_shape)
        obs_epistemic_var = torch.var(obs_means, dim=1)

        # r_t reward for each timestep: (self.plan_hor, popsize)
        r_t = (1 / self.dO) * torch.sum(torch.sqrt(obs_epistemic_var / w_base), dim=-1)
        # costs: (popsize,) summed cost over all timesteps for each ac seq
        costs = -torch.sum(r_t, dim=0)
        # Replace nan with high cost
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

    @torch.no_grad()
    def _compile_cost(self, ac_seqs):
        return (self._compile_cost_reward(ac_seqs) + self._compile_cost_intrinsic(ac_seqs)) / 2.0

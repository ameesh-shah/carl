from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import trange

import numpy as np
import torch

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from explore_base_mpc import ExploreMPC
from env.pointmass import PointmassEnv

class ExploreEnsembleVarianceMPC(ExploreMPC):

    def __init__(self, params):
        super().__init__(params)

        # Override cost function to be exploration reward
        self.optimizer.cost_function = self._compile_cost

    @torch.no_grad()
    def _compile_cost_intrinsic(self, ac_seqs, cur_obs):
        """Computes intrinsic exploration cost (negated exploration_reward). Incentivize agents to visit states
        with high epistemic uncertainty (high variance between bootstrap predictions of next state).

        Args:
            ac_seqs (torch.Tensor):
            cur_obs (torch.Tensor):
        Returns:
            cost (ndarray): 
        """
        obs_vars = []
        obs_means = []

        for t in range(self.plan_hor):
            cur_acs = ac_seqs[t]

            # next_obs shape: (npart * pop_size, obs_shape) = (8000, 4)
            # mean, var shape: (num_nets, npart * popsize / num_nets, obs_shape) = (5, 1600, 4)
            # calculate variance over all bootstraps
            next_obs, (mean, var) = self._predict_next_obs(cur_obs, cur_acs, return_mean_var=True)

            # each of `popsize` CEM samples is a different action, so we shouldn't avg states over popsize
            # mean: (num_nets, npart / num_nets, popsize, env obs_shape [without extra obs like catastrophe_prob])
            mean = mean.view(self.model.num_nets, self.npart // self.model.num_nets,
                    self.optimizer.popsize, -1)
            # obs_mean_per_bootstrap: (num_nets, popsize, obs_shape)
            # average next state prediction (over all particles) for each bootstrap
            obs_mean_per_bootstrap = torch.mean(mean, dim=1)

            obs_means.append(obs_mean_per_bootstrap)
            obs_vars.append(var)

            cur_obs = self.obs_postproc2(next_obs)

        # mean and var in obs predictions over all particles, for each sample in `popsize`, according to each bootstrap
        # Calculate max aleatoric var for each state component to standardize
        obs_vars = torch.stack(obs_vars)
        assert obs_vars.shape[:-1] == (self.plan_hor, self.model.num_nets,
            (self.npart * self.optimizer.popsize) // self.model.num_nets)
        # obs_vars: (self.plan_hor, num_net, npart * popsize / num_nets, obs_shape) -> (-1, obs_shape)
        env_dO = obs_vars.shape[-1]
        obs_vars = obs_vars.view(-1, env_dO)
        # w_base: (obs_shape,)
        w_base, _ = torch.max(obs_vars, dim=0)

        # Calculate variance over bootstrap mean predictions
        # obs_means: (self.plan_hor, num_nets, popsize, env obs_shape)
        obs_means = torch.stack(obs_means)
        assert obs_means.shape == (self.plan_hor, self.model.num_nets, self.optimizer.popsize, env_dO)
        # Disagreement (var) across bootstraps about the next state indicates epistemic uncertainty
        # obs_epistemic_var: (self.plan_hor, popsize, obs_shape)
        obs_epistemic_var = torch.var(obs_means, dim=1)

        # r_t reward for each timestep: (self.plan_hor, popsize)
        # reward agent for visiting states with high epistemic uncertainty
        r_t = (1 / env_dO) * torch.sum(torch.sqrt(obs_epistemic_var / w_base), dim=-1)
        # costs: (popsize,) summed cost over all timesteps for each ac seq
        costs = -torch.sum(r_t, dim=0)
        # Replace nan with high cost
        costs[costs != costs] = 1e6
        plot_tensors = {
                "costs_per_step": costs
                }
        return costs.detach().cpu().numpy()

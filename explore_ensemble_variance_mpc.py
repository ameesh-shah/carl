from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import trange

import torch

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from MPC import MPC


class ExploreEnsembleVarianceMPC(MPC):

    def __init__(self, params):
        super().__init__(params)

        # Override cost function to be exploration reward
        self.optimizer.cost_function = self._compile_cost

    def _predict_next_obs(self, obs, acs, return_mean_var=False):
        proc_obs = self.obs_preproc(obs)

        assert self.prop_mode == 'TSinf'
        proc_obs = self._expand_to_ts_format(proc_obs)
        acs = self._expand_to_ts_format(acs)

        inputs = torch.cat((proc_obs, acs), dim=-1)

        mean, var, catastrophe_prob = self.model(inputs)

        predictions = mean + torch.randn_like(mean, device=TORCH_DEVICE) * var.sqrt()

        predictions = torch.cat((predictions, catastrophe_prob), dim=-1)

        # TS Optimization: Remove additional dimension
        predictions = self._flatten_to_matrix(predictions)

        if return_mean_var:
            return self.obs_postproc(obs, predictions), (mean, var)

        return self.obs_postproc(obs, predictions)


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
        cur_obs = cur_obs[None]
        cur_obs = cur_obs.expand(self.optimizer.popsize * self.npart, -1)

        intrinsic_cost = self._compile_cost_intrinsic(ac_seqs, cur_obs)
        supervised_cost = self._compile_cost_reward(ac_seqs, cur_obs)

        # TODO: make weight on each a parameter
        return (intrinsic_cost + supervised_cost) / 2.0

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
        costs = torch.zeros(self.optimizer.popsize, self.npart, device=TORCH_DEVICE)
        
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
        mean_cost = costs.mean()
        return mean_cost.detach().cpu().numpy()

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
            # mean, var shape: (num_nets, npart * popsize / num_nets, obs_shape) = (5, 1600, 4)
            # calculate variance over all bootstraps
            next_obs, (mean, var) = self._predict_next_obs(cur_obs, cur_acs, return_mean_var=True)
            # cost shape: TODO
            cost = self.obs_cost_fn(next_obs) + self.ac_cost_fn(cur_acs)
            if self.mode == 'test' and not self.no_catastrophe_pred: # use catastrophe prediction during adaptation
                # catastrophe_cost_fn masks `cost`
                #   if there is a catastrophe, the cost for that timestep is increased by COLLISION_COST=1e4 (configured in config/{env}.py)
                #   else, the cost is `cost`
                # self.percentile (default 100 during train / 50 during test) controls when we mark a state as trajectory
                #   e.g. if 50, we get marked as catastrophe if (predicted) next_obs catastrophe >= .5
                #   setting it lower results in more risk-averse planning (we avoid states if there is even a small prob of catastrophe)
                cost = self.catastrophe_cost_fn(next_obs, cost, self.percentile)
            elif self.mode == 'train' and self.unsafe_pretraining:
                catastrophe_prob = next_obs[..., -1]
                cost = -(100 * catastrophe_prob) # negate so cost is in [-100, 0] (lowest cost for catastrophe_prob=1)
            cost = cost.view(-1, self.npart)
            costs += cost
            cur_obs = self.obs_postproc2(next_obs)

        # Replace nan with high cost
        costs[costs != costs] = 1e6
        mean_cost = costs.mean(dim=1)
        return mean_cost.detach().cpu().numpy()

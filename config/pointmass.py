from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DotmapUtils import get_required_argument
from config.ensemble_model import EnsembleModel

import gym
import numpy as np
import torch

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PointmassConfigModule:
    ENV_NAME = "" # FILL IN SUBCLASS
    TASK_HORIZON = 150
    NTRAIN_ITERS = 25
    NROLLOUTS_PER_ITER = 1
    NTEST_ROLLOUTS = 1
    PLAN_HOR = 2
    MODEL_IN, MODEL_OUT = 4, 2
    """
    MODEL_IN: (x, y, ac_x, ac_y) 
    MODEL_OUT (x, y)
    """
    GP_NINDUCING_POINTS = 200
    CATASTROPHE_SIGMOID = torch.nn.Sigmoid()
    CATASTROPHE_COST = 10000
    MODEL_ENSEMBLE_SIZE = 5
    MODEL_HIDDEN_SIZE = 500
    MODEL_WEIGHT_DECAYS = [1e-4, 2.5e-4, 2.5e-4, 5e-4]
    USE_DENSE_REWARD = True

    # Create and move this tensor to GPU so that
    # we do not waste time moving it repeatedly to GPU later
    ee_sub = torch.tensor([0.0, 0.6], device=TORCH_DEVICE, dtype=torch.float)

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        self.NN_TRAIN_CFG = {"epochs": 5}
        # FIXME: this should be DCEM, but seems unrelated to
        # how the optimizer is actually set up. Ignore this
        # for now.
        self.OPT_CFG = {
            "CEM": {
                "popsize": 400,
                "num_elites": 40,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    @staticmethod
    def obs_preproc(obs):
        """Preprocesses obs to construct input for the model.
        Args:
            obs: [xpos, ypos, goalx, goaly, catastrophe] (see half-cheetah for a similar reference)
        Returns:
            [xpos, ypos]
        """
        assert obs.shape[-1] == 5
        return obs[..., :2]

    # This post-processing function is used in _predict_next_obs in MPC,
    # which will subsequently be used in _compile_cost.
    # 
    # The function prepares a next state from model prediction.
    # Some processing steps have been done in the body of _predict_next_obs,
    # here we just apply a sigmoid function to the catastrophe_prob.
    # Also insert the current action_noise (obtained from obs) at index 2.
    @staticmethod
    def obs_postproc(obs, pred):
        """Post processes obs to prepare next_obs for the next iteration (i.e. calculate the
        next state from the predicted state diff in this case and add back the additional info
        we carry around in obs)."""
        pred_state_change = pred[..., :2] # remove catastrophe_state dim and other extra info
        pred_next_state = obs[..., :2] + pred_state_change 
        goal_coords = obs[..., 2:4]
        return torch.cat((
            pred_next_state,
            goal_coords,
            CONFIG_MODULE.CATASTROPHE_SIGMOID(pred[..., -1:])), dim=-1) 

    @staticmethod
    def targ_proc(obs, next_obs):
        """Constructs target for training model.

        Returns: (note we do NOT predict pendulum len)
            ((normalized) state delta, catastrophe_prob) 
        """
        state_delta = next_obs[..., :2] - obs[..., :2]
        next_catastrophe_prob = next_obs[...,-1:]
        return np.concatenate((state_delta, next_catastrophe_prob), axis=-1)

    @staticmethod
    def obs_cost_fn(obs):
        """Gets the reward dim we conveniently store in obs. 
        Args:
            obs: shape (batch_size, obs_dim) = (npart * popsize, obs_dim) = (8000, ...)
        """
        # TODO: this is dense reward specifically
        # print("next obs: ", obs[:, :2])
        return torch.norm(obs[:, :2] - obs[:, 2:4], dim=-1)

    @staticmethod
    def ac_cost_fn(acs):
        return torch.zeros(acs.shape[0], device=TORCH_DEVICE)

    @staticmethod
    def catastrophe_cost_fn(obs, cost, percentile):
        catastrophe_mask = obs[..., -1] > percentile / 100
        cost[catastrophe_mask] += CONFIG_MODULE.CATASTROPHE_COST
        return cost

    def nn_constructor(self, model_init_cfg):

        ensemble_size = get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size")

        load_model = model_init_cfg.get("load_model", False)

        assert load_model is False, 'Has yet to support loading model'

        model = EnsembleModel(ensemble_size,
                        in_features=self.MODEL_IN,
                        out_features=self.MODEL_OUT * 2 + 1, 
                        hidden_size=self.MODEL_HIDDEN_SIZE,
                        num_layers=len(self.MODEL_WEIGHT_DECAYS),
                        weight_decays=self.MODEL_WEIGHT_DECAYS).to(TORCH_DEVICE)

        model.optim = torch.optim.Adam(model.parameters(), lr=0.001)

        return model


CONFIG_MODULE = PointmassConfigModule

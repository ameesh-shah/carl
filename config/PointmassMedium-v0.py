from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from DotmapUtils import get_required_argument
from config.ensemble_model import EnsembleModel

import gym
import numpy as np
import torch

TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PointmassMediumConfigModule:
    ENV_NAME = "PointmassMedium-v0"
    TASK_HORIZON = 150
    NTRAIN_ITERS = 50
    NROLLOUTS_PER_ITER = 1
    NTEST_ROLLOUTS = 1
    PLAN_HOR = 25
    MODEL_IN, MODEL_OUT = 4, 3 # In; (x, y, ac_x, ac_y), Out: (x, y, prob)
    GP_NINDUCING_POINTS = 200
    CATASTROPHE_SIGMOID = torch.nn.Sigmoid()
    CATASTROPHE_COST = 10000
    MODEL_ENSEMBLE_SIZE = 5
    MODEL_HIDDEN_SIZE = 500
    MODEL_WEIGHT_DECAYS = [1e-4, 2.5e-4, 2.5e-4, 5e-4]

    # Create and move this tensor to GPU so that
    # we do not waste time moving it repeatedly to GPU later
    ee_sub = torch.tensor([0.0, 0.6], device=TORCH_DEVICE, dtype=torch.float)

    def __init__(self):
        self.ENV = gym.make(self.ENV_NAME)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "CEM": {
                "popsize": 400,
                "num_elites": 40,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    # Pre-process the obs so that it can be fed into the ensemble
    @staticmethod
    def obs_preproc(obs):
        # ..., action_noise, catastrophe
        # The function simply removes the action_noise
        # print(obs)
        # print(obs.shape)
        assert obs.shape[-1] == 4
        if isinstance(obs, np.ndarray):
            return obs[..., :-2]
        else:
            return obs[..., :-2]
        return obs

    # This post-processing function is used in _predict_next_obs in MPC,
    # which will subsequently be used in _compile_cost.
    # 
    # The function prepares a next state from model prediction.
    # Some processing steps have been done in the body of _predict_next_obs,
    # here we just apply a sigmoid function to the catastrophe_prob
    @staticmethod
    def obs_postproc(obs, pred):
        return torch.cat((pred[..., :-1], CONFIG_MODULE.CATASTROPHE_SIGMOID(pred[..., -1:])), dim=-1) 

    # This function prepares ground truth next states and catastrophe prob.
    # In MPC, we see "self.targ_proc(obs[:-1], obs[1:])" being used. 
    # FIXME: double check that the target should be 
    # batch * [next_x, next_y, catastrophe_prob]
    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs

    # FIXME: instead of using this, use reward from step().
    @staticmethod
    def obs_cost_fn(obs):
        return 0

    @staticmethod
    def ac_cost_fn(acs):
        return 0

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

    @staticmethod
    def catastrophe_cost_fn(obs, cost, percentile):
        """
        print('***** obs')
        print(obs.shape)
        print(obs)
        print('***** cost')
        print(cost)
        print('***** percentile')
        print(percentile)
        """
        catastrophe_mask = obs[..., -1] > percentile / 100
        
        """
        print('***** catastroophe mask')
        print(catastrophe_mask)
        print('***** obs[..., -1]')
        print(obs[..., -1])
        for i in obs[..., -1]:
            if i > 0:
                print(i)
        """

        """
        print(type(catastrophe_mask))
        print(catastrophe_mask.shape)
        print(type(cost))
        print(cost.shape)
        """
        print('before catas mask')
        print(cost)
        cost[catastrophe_mask] += CONFIG_MODULE.CATASTROPHE_COST
        print(cost)
        """
        print('***** cost[catastrophe_mas]')
        print(cost[catastrophe_mask])
        # exit()
        """

        return cost

CONFIG_MODULE = PointmassMediumConfigModule

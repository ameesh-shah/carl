from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
from tqdm import trange

import torch
TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Agent:
    """An general class for RL agents.
    """

    def sample(self, horizon, policy, record=False, env=None, mode='train'):
        """Samples a rollout from the agent.

        Arguments:
            horizon: (int) The length of the rollout to generate from the agent.
            policy: (policy) The policy that the agent will use for actions.
            record: (bool) Whether to record the rollout
            env: (gym.Env) Environment to rollout
            mode: (str) Whether we're training or testing, used for resetting the env

        Returns: (dict) A dictionary containing data from the rollout.
        """
        times, rewards = [], []
        catastrophes = 0
        policy.mode = mode
        # env.reset selects a random environment with the train/test distribution
        O, A, reward_sum, done = [env.reset(mode=mode)], [], 0, False
        policy.reset()
        for t in trange(horizon):
            start = time.time()
            policy_action = policy.act(O[t], t)
            A.append(policy_action)
            times.append(time.time() - start)
            obs, reward, done, info = env.step(policy_action) # A[t]
            # print("=== taking action ", policy_action) 
            # print("cur obs: ", O[t]) 
            # print("actual next obs: ", obs)

            O.append(obs)
            reward_sum += reward
            rewards.append(reward)

            # Run through model to see the catastrophe prob
            proc_obs = policy.obs_preproc(O[t])
            inputs = torch.cat((torch.Tensor(proc_obs).to(TORCH_DEVICE), torch.Tensor(policy_action).to(TORCH_DEVICE)), dim=-1)

            # Getting catastrophe prob for a state
#            import pdb; pdb.set_trace()
            mean, var, catastrophe_prob = policy.model(inputs)
            print("CAT PROB: ", catastrophe_prob)

            catastrophe_prob[catastrophe_prob != catastrophe_prob] = 0.
            _catastrophe_prob = torch.mean(catastrophe_prob)
            _catastrophe_prob = torch.sigmoid(_catastrophe_prob)

            env.catastrophe_probs.append(_catastrophe_prob.detach().cpu().numpy())

            if info['Catastrophe']:
                import pdb; pdb.set_trace()
                catastrophes += 1

            if done:
                break
        if record:
            env.close()
        print("Average action selection time: ", np.mean(times))
        print("Rollout length: ", len(A))

        # (resolved) a rather strange bug. In Pointmass env, np.array(O)
        # returns an ndarray of shape (151,), i.e. ndim=1. But in cartpole, the same
        # code returns an ndarray of shape (201, 6), i.e. ndim=2.
        # 
        # env.reset() in pointmass did not return extended state, which caused the
        # 1st element of the list to have a different shape than others.
        # Ndarray therefore did not interpret the 2nd dimension as part of its shape.

        return {
            "obs": np.array(O),
            "ac": np.array(A),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards),
            "catastrophe": catastrophes,
        }

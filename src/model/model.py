""" Trajectory model prototype. 
    This file is kept as a template,
    all models should implement the same interface.
"""

import numpy as np
import torch
import torch.nn as nn


class TrajectoryModel(nn.Module):

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_size=64,
        max_ep_len=100,
        action_type="continuous",
    ):
        """
        :param obs_dim: observation dimension (as a flattened tensor)
        :param act_dim: action dimension (as a flattened tensor)
        :param task_num: number of tasks to be encoded
        :param hidden_size: the dimension of the embedding space
        :param max_ep_len: (Optional) maximal episode length
        """

        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_ep_len = max_ep_len
        self.action_type = action_type

    def forward(
        self,
        obs,
        actions,
        rewards,
        current_obs=None,
        current_action=None,
        current_reward=None,
        attention_mask=None,
        step_ids=None,
        current_step_id=None,
    ):

        return None, None, None

    def get_action(
        self,
        tasks,
        obs,
        actions,
        rewards,
        timesteps,
        current_obs=None,
        current_step_id=None,
        **kwargs
    ):

        return None

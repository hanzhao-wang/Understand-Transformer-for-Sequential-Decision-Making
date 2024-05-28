import numpy as np
import torch
import torch.nn as nn
import transformers

from model.model import TrajectoryModel

"""
The code is based on the below repository,
https://github.com/CarperAI/Algorithm-Distillation-RLHF/blob/main/algorithm_distillation/models/gpt2.py
"""


class GPT2_OM(TrajectoryModel):
    def __init__(
        self,
        obs_dim,
        act_value_dim,
        output_act_dim,
        hidden_size,
        max_ep_len=100,
        tar_length=100,
        need_time_embed=False,
        mask_act_idx=True,
        **kwargs
    ):
        super().__init__(obs_dim, act_value_dim, hidden_size, max_ep_len)

        self.output_act_dim = output_act_dim
        self.need_time_embed = need_time_embed

        # The embedding layers for obs/action/reward.
        self.obs_embedding = torch.nn.Linear(self.obs_dim, self.hidden_size)
        self.act_embedding = torch.nn.Linear(self.act_dim, self.hidden_size)
        self.act_id_embedding = torch.nn.Embedding(
            self.output_act_dim, self.hidden_size
        )
        self.rew_embedding = torch.nn.Linear(1, self.hidden_size)
        self.tar_length = tar_length
        # Generate the most basic GPT2 config
        config = transformers.GPT2Config(
            vocab_size=1, n_embd=hidden_size, **kwargs
        )
        self.transformers = transformers.GPT2Model(config)
        # This is our time embedding based on steps t.
        self.step_embedding = torch.nn.Embedding(max_ep_len, self.hidden_size)

        # The last layers mapping to obs/act_id/action/reward from the embedding space if needed.
        self.obs_head = torch.nn.Linear(self.hidden_size, self.obs_dim)
        self.act_head = torch.nn.Linear(self.hidden_size, self.output_act_dim)
        self.act_value_head = torch.nn.Linear(self.hidden_size, self.act_dim)
        self.rew_head = torch.nn.Linear(self.hidden_size, 1)

        self.layer_norm_in_embedding = torch.nn.LayerNorm(self.hidden_size)
        self.output_attention = False
        self.output_hidden_states = False
        self.need_probs = False
        self.mask_act_idx = mask_act_idx

    def forward(
        self,
        obs,
        act_idx,
        act_values,
        rewards,
        current_obs=None,
        current_act_id=None,
        current_action=None,
        current_reward=None,
        attention_mask=None,
        step_ids=None,
        current_step_id=None,
    ):
        device = obs.device
        # current_task=tasks_idxs.clone()

        batch_size, tar_horizon, _ = obs.shape
        # extend the task to the same length as obs, actions, rewards

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones(
                (batch_size, tar_horizon), dtype=torch.float, device=device
            )

        if step_ids is None:
            step_ids = torch.arange(
                0, tar_horizon, dtype=torch.long, device=device
            ).view(batch_size, tar_horizon)
        embedded_steps = self.step_embedding(step_ids).view(
            batch_size, tar_horizon, self.hidden_size
        )
        if self.need_time_embed == False:
            embedded_steps = torch.tensor(0)

        embedded_obs = self.obs_embedding(obs) + embedded_steps
        embedded_act_id = self.act_id_embedding(act_idx) + embedded_steps
        embedded_act = self.act_embedding(act_values) + embedded_steps
        embedded_rew = self.rew_embedding(rewards) + embedded_steps

        if current_step_id is None or self.need_time_embed == False:
            embedded_latest_step = torch.tensor(0)
        else:
            embedded_latest_step = self.step_embedding(current_step_id)

        # Embed the current obs/action/reward (stop if None)
        latest = [current_obs, current_act_id, current_action, current_reward]
        apply_cls = [
            self.obs_embedding,
            self.act_id_embedding,
            self.act_embedding,
            self.rew_embedding,
        ]
        extra = []
        for inp, cls in zip(latest, apply_cls):
            if inp is not None:
                extra.append(
                    cls(inp) + embedded_latest_step
                )  # (b,1, hidden_size)
            else:
                break
        num_extra = len(extra)  # number of non-empty current obs/action/reward
        extra = (
            None if num_extra == 0 else torch.cat(extra, dim=1)
        )  # (b, i, hidden_size)

        # Stack the input into (obs, act, rew, obs, act, rew, ...) sequence.
        # Note: only affects axis 1. Axis 0 (batch) and axis 2 (embedding) are preserved.
        input_seq = stack_seq(
            embedded_obs, embedded_act_id, embedded_act, embedded_rew, extra
        )
        input_seq = self.layer_norm_in_embedding(input_seq)

        attention_mask = (
            attention_mask.unsqueeze(-1).repeat((1, 1, 4)).view(batch_size, -1)
        )  # (b, 4*t)
        if self.mask_act_idx:  # mask out the action index input
            attention_mask[:, 1::4] = 0

        attention_mask = torch.concat(
            [
                attention_mask,
                torch.ones(
                    (batch_size, num_extra), dtype=torch.float, device=device
                ),
            ],
            dim=1,
        )

        # Do inference using the underlying transformer.
        output = self.transformers(
            inputs_embeds=input_seq,
            attention_mask=attention_mask,
            output_attentions=self.output_attention,
            output_hidden_states=self.output_hidden_states,
        )  # (b, *, hidden_size)
        pred_act_ids = output["last_hidden_state"][
            :, ::4, :
        ]  # (b, t, hidden_size) based on last state. We only predict the idx as the actions since in the continuous action space we don't need the idx (which is always 0)
        pred_act_values = output["last_hidden_state"][:, 1::4, :]
        pred_rewards = output["last_hidden_state"][
            :, 2::4, :
        ]  # (b, t, hidden_size)
        pred_obs = output["last_hidden_state"][
            :, 3::4, :
        ]  # (b, t, hidden_size)

        if self.output_attention:
            return (
                self.act_head(pred_act_ids),
                self.rew_head(pred_rewards),
                output["attentions"],
            )
        elif self.output_hidden_states:
            return (
                self.act_head(pred_act_ids),
                self.rew_head(pred_rewards),
                output.hidden_states,
            )
        else:
            return self.act_head(pred_act_ids), self.rew_head(pred_rewards)

    def get_action(
        self,
        obs,
        act_ids,
        act_values,
        rewards,
        timesteps,
        current_obs=None,
        current_step_id=None,
        temp=1,
        **kwargs
    ):
        """
        :param tasks: (b,1)
        :param obs: (b, t, obs_dim)
        :param actions: (b, t, act_dim)
        :param rewards: (b, t, 1)
        :param timesteps: (b, t)
        :param current_obs: (Optional) (b, obs_dim)
        :param current_step_id: (Optional) the latest step id applied to the latest obs. (b,1)
        return the predicted action at the current step (b,1)
        """

        if (
            self.tar_length is not None
        ):  # truncate to tar length, where care the recent tar_length steps
            obs = obs[:, -self.tar_length :, :]
            act_ids = act_ids[:, -self.tar_length :]
            act_values = act_values[:, -self.tar_length :, :]
            rewards = rewards[:, -self.tar_length :, :]
            timesteps = timesteps[:, -self.tar_length :]

            # pad all tokens to sequence length。 All tokens are padded at the beginning of the sequence
            attention_mask = torch.cat(
                [
                    torch.zeros(self.tar_length - obs.shape[1]),
                    torch.ones(obs.shape[1]),
                ]
            )
            attention_mask = attention_mask.to(
                dtype=torch.long, device=obs.device
            ).reshape(1, -1)
            obs = torch.cat(
                [
                    torch.zeros(
                        (
                            obs.shape[0],
                            self.tar_length - obs.shape[1],
                            self.obs_dim,
                        ),
                        device=obs.device,
                    ),
                    obs,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            act_ids = torch.cat(
                [
                    torch.zeros(
                        (act_ids.shape[0], self.tar_length - act_ids.shape[1]),
                        device=act_ids.device,
                    ),
                    act_ids,
                ],
                dim=1,
            ).to(dtype=torch.long)
            act_values = torch.cat(
                [
                    torch.zeros(
                        (
                            act_values.shape[0],
                            self.tar_length - act_values.shape[1],
                            self.act_dim,
                        ),
                        device=act_values.device,
                    ),
                    act_values,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            rewards = torch.cat(
                [
                    torch.zeros(
                        (
                            rewards.shape[0],
                            self.tar_length - rewards.shape[1],
                            1,
                        ),
                        device=rewards.device,
                    ),
                    rewards,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (
                            timesteps.shape[0],
                            self.tar_length - timesteps.shape[1],
                        ),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        if self.output_act_dim > 1:  # discrete action case
            results = self.forward(
                obs,
                act_ids,
                act_values,
                rewards,
                current_obs,
                None,
                None,
                None,
                attention_mask,
                timesteps,
                current_step_id,
            )
            return_preds = results[1]
            action_preds = results[0]
            probs = (
                torch.nn.functional.softmax(
                    action_preds[:, -1, :].reshape(-1) / temp, dim=-1
                )
                .cpu()
                .detach()
                .numpy()
            )
            sample_action = np.random.choice(self.output_act_dim, 1, p=probs)
            if len(results) > 2:
                return (
                    torch.tensor(sample_action),
                    return_preds[:, -1, :],
                    results[2],
                )
            elif self.need_probs:
                return (
                    torch.tensor(sample_action),
                    return_preds[:, -1, :],
                    probs,
                )
            else:
                return torch.tensor(sample_action), return_preds[:, -1, :]
        else:  # cts action case
            results = self.forward(
                obs,
                act_ids,
                act_values,
                rewards,
                current_obs,
                None,
                None,
                None,
                attention_mask,
                timesteps,
                current_step_id,
            )
            return_preds = results[1]
            action_preds = results[0]
            if len(results) > 3:
                return (
                    action_preds[:, -1, :],
                    return_preds[:, -1, :],
                    results[2],
                )

            else:
                return action_preds[:, -1, :], return_preds[:, -1, :]

    def get_return(
        self,
        obs,
        act_ids,
        act_values,
        rewards,
        timesteps,
        current_obs,
        current_step_id,
        current_act_id,
        current_action,
        **kwargs
    ):

        if (
            self.tar_length is not None
        ):  # truncate to tar length, where care the recent tar_length steps
            obs = obs[:, -self.tar_length :, :]
            act_ids = act_ids[:, -self.tar_length :]
            act_values = act_values[:, -self.tar_length :, :]
            rewards = rewards[:, -self.tar_length :, :]
            timesteps = timesteps[:, -self.tar_length :]

            # pad all tokens to sequence length。 All tokens are padded at the beginning of the sequence
            attention_mask = torch.cat(
                [
                    torch.zeros(self.tar_length - obs.shape[1]),
                    torch.ones(obs.shape[1]),
                ]
            )
            attention_mask = attention_mask.to(
                dtype=torch.long, device=obs.device
            ).reshape(1, -1)
            obs = torch.cat(
                [
                    torch.zeros(
                        (
                            obs.shape[0],
                            self.tar_length - obs.shape[1],
                            self.obs_dim,
                        ),
                        device=obs.device,
                    ),
                    obs,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            act_ids = torch.cat(
                [
                    torch.zeros(
                        (act_ids.shape[0], self.tar_length - act_ids.shape[1]),
                        device=act_ids.device,
                    ),
                    act_ids,
                ],
                dim=1,
            ).to(dtype=torch.long)
            act_values = torch.cat(
                [
                    torch.zeros(
                        (
                            act_values.shape[0],
                            self.tar_length - act_values.shape[1],
                            self.act_dim,
                        ),
                        device=act_values.device,
                    ),
                    act_values,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            rewards = torch.cat(
                [
                    torch.zeros(
                        (
                            rewards.shape[0],
                            self.tar_length - rewards.shape[1],
                            1,
                        ),
                        device=rewards.device,
                    ),
                    rewards,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (
                            timesteps.shape[0],
                            self.tar_length - timesteps.shape[1],
                        ),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        results = self.forward(
            obs,
            act_ids,
            act_values,
            rewards,
            current_obs,
            current_act_id,
            current_action,
            None,
            attention_mask,
            timesteps,
            current_step_id,
        )
        return_preds = results[1]

        return return_preds[:, -1, :]


def stack_seq(contexts, act_ids, act_values, rew, extra=None) -> torch.Tensor:
    """
    Stack up into a sequence (contexts, act_id, act_value, rew, contexts, act_id, cat_value, rew, ...) in axis 1,
    and append extra in the end.
    :param contexts: shape (b, t, hidden_size)
    :param act: shape (b, t, hidden_size)
    :param rew: shape (b, t, hidden_size)
    :param extra: (Optional) shape (b, i, hidden_size) where i can be 1, 2, 3,4
    :return: shape (b, 4*t+i, hidden_size)
    """
    batch_size, tar_horizon, _ = contexts.shape
    stacked = (
        torch.stack((contexts, act_ids, act_values, rew), dim=1)
        .permute(0, 2, 1, 3)
        .reshape(batch_size, 4 * tar_horizon, -1)
    )
    if extra is None:
        return stacked
    else:
        return torch.concat([stacked, extra], dim=1)

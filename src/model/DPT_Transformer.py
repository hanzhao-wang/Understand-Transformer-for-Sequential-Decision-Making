""" Reimplementation of Decision-Pretrained Transformer (DPT).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


class DecisionPretrainedTransformer(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_value_dim: int,
        output_act_dim: int,
        action_type: str,
        hidden_size: int,
        n_layer: int,
        n_head: int,
        n_inner: int,
        resid_pdrop: float,
        attn_pdrop: float,
        max_ep_len: int = 100,
        tar_length: int = 100,
        need_time_embed: bool = False,
        mask_act_idx: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_value_dim = act_value_dim
        self.output_act_dim = output_act_dim
        self.action_type = action_type
        self.hidden_size = hidden_size
        self.max_ep_len = max_ep_len
        self.tar_length = tar_length
        self.need_time_embed = need_time_embed
        self.mask_act_idx = mask_act_idx
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.model_type = "DPT"

        # gpt2 config
        config = transformers.GPT2Config(
            n_positions=4 * (1 + self.max_ep_len),
            n_embd=self.hidden_size,
            n_layer=self.n_layer,
            n_head=1,  # ! yes this is specified in original implementation
            resid_pdrop=self.resid_pdrop,
            attn_pdrop=self.attn_pdrop,
            embd_pdrop=self.resid_pdrop,
            use_cache=False,
        )
        self.transformer = transformers.GPT2Model(config)

        self.embed_transition = nn.Linear(
            2 * self.obs_dim + self.act_value_dim + 1, self.hidden_size
        )
        self.pred_action = nn.Linear(self.hidden_size, self.output_act_dim)

    def forward(
        self,
        contexts,
        act_idx,
        act_values,
        feedbacks,
        current_ctx=None,
        current_act_idx=None,
        current_action=None,
        current_feedback=None,
        attention_mask=None,
        step_ids=None,
        current_step_id=None,
    ):
        device = contexts.device
        batch_size = contexts.shape[0]
        if current_ctx is None:
            # ! use last state as query state
            query_state = contexts[:, -1, :][:, None, :]
            zeros = torch.zeros(
                batch_size, 1, self.obs_dim + self.act_value_dim + 1
            ).to(device)

            state_seq = torch.cat([query_state, contexts[:, :-1, :]], dim=1)
            action_seq = torch.cat(
                [zeros[:, :, : self.act_value_dim], act_values[:, :-1, :]],
                dim=1,
            )
            next_state_seq = torch.cat(
                [zeros[:, :, : self.obs_dim], contexts[:, 1:, :]], dim=1
            )
            reward_seq = torch.cat(
                [zeros[:, :, :1], feedbacks[:, :-1, :]], dim=1
            )
        else:
            # ! use current_ctx as query state - also the latest state
            # ! query_state/current_ctx is 3-dimensional
            # query_state = current_ctx[:, None, :]
            query_state = current_ctx.to(device)
            zeros = torch.zeros(
                batch_size, 1, self.obs_dim + self.act_value_dim + 1
            ).to(device)
            state_seq = torch.cat([query_state, contexts], dim=1)
            action_seq = torch.cat(
                [zeros[:, :, : self.act_value_dim], act_values],
                dim=1,
            )
            next_state_seq = torch.cat(
                [zeros[:, :, : self.obs_dim], contexts[:, 1:, :], query_state],
                dim=1,
            )[:, : state_seq.shape[1], :]
            reward_seq = torch.cat(
                [zeros[:, :, :1], feedbacks],
                dim=1,
            )

        tar_horizon = state_seq.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, tar_horizon), dtype=torch.long, device=device
            )

        # print(
        #     state_seq.shape,
        #     action_seq.shape,
        #     next_state_seq.shape,
        #     reward_seq.shape,
        # )

        seq = torch.cat(
            [state_seq, action_seq, next_state_seq, reward_seq], dim=2
        )
        stacked_inputs = self.embed_transition(seq)
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        preds = self.pred_action(transformer_outputs["last_hidden_state"])
        # ! note that here we do not need to use preds[:, 1:, :]
        # ! because we use last state as query state
        # ! return -1 as placeholder for pred_reward

        return preds, -torch.ones(
            (batch_size, preds.shape[1], 1), device=device
        )

    def get_action(
        self,
        contexts,
        act_ids,
        acts,
        feedbacks,
        timesteps=None,
        current_ctx=None,
        current_step_id=None,
        temp=1,
        **kwargs
    ):
        if self.tar_length is not None:
            contexts = contexts[:, : self.tar_length, :]
            acts = acts[:, : self.tar_length, :]
            feedbacks = feedbacks[:, : self.tar_length, :]

        # ! note that we do not make an explicit attn_mask here
        # ! because the actual input_seq length is 1 + tar_length

        # print(contexts.shape, acts.shape, feedbacks.shape, current_ctx.shape)

        preds = self.forward(
            contexts,
            None,
            acts,
            feedbacks,
            current_ctx,
        )

        rwd_preds = preds[1]
        act_preds = preds[0]

        if self.action_type == "discrete":
            probs = (
                torch.nn.functional.softmax(act_preds[:, -1, :] / temp, dim=-1)
                .cpu()
                .detach()
                .numpy()
            )
            sample_actions = [
                np.random.choice(self.output_act_dim, p=prob) for prob in probs
            ]
            return torch.tensor(sample_actions), rwd_preds[:, -1, :]
        else:
            return act_preds[:, -1, :], rwd_preds[:, -1, :]

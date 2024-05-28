import numpy as np
import torch
import torch.nn as nn
import transformers

from model.model import TrajectoryModel


class GPT2_ICL(TrajectoryModel):
    def __init__(
        self,
        obs_dim,
        act_value_dim,
        output_act_dim,
        action_type,
        hidden_size,
        max_ep_len=100,
        tar_length=100,
        need_time_embed=False,  # placeholder
        mask_act_idx=True,  # placeholder
        **kwargs
    ):
        super().__init__(
            obs_dim, act_value_dim, hidden_size, max_ep_len, action_type
        )

        self.output_act_dim = output_act_dim
        self.model_type = "ICL"

        # The embedding layers for context+1/action.
        self.obs_embedding = torch.nn.Linear(
            self.obs_dim + 1, self.hidden_size
        )  # assume the feedback is 1-dim
        self.act_embedding = torch.nn.Linear(
            self.act_dim, self.hidden_size
        )  # the input action has dimension act_dim
        self.tar_length = tar_length
        # Generate the most basic GPT2 config
        config = transformers.GPT2Config(
            vocab_size=1, n_embd=hidden_size, **kwargs
        )
        self.transformers = transformers.GPT2Model(config)

        # The last layers mapping to contexts/action/ from the embedding space if needed.
        self.obs_head = torch.nn.Linear(self.hidden_size, 1)
        self.act_head = torch.nn.Linear(self.hidden_size, self.output_act_dim)

        self.layer_norm_in_embedding = torch.nn.LayerNorm(self.hidden_size)
        self.output_attention = False
        self.output_hidden_states = False
        self.need_probs = False
        self.same_embedding = False

    def forward(
        self,
        contexts,
        act_idx,  # not used here just for place holder
        act_values,
        feedbacks,
        current_ctx=None,
        current_act_id=None,  # not used here just for place holder
        current_action=None,
        current_feedback=None,  # not used here just for place holder
        attention_mask=None,
        step_ids=None,  # not used here just for place holder
        current_step_id=None,  # not used here just for place holder
    ):
        """
        :param contexts: (b, t, ctx_dim)
        :param act_values: (b, t, act_dim)
        :param feedbacks: (b, t, 1)
        :param current_ctx= (Optional) (b,1, obs_dim)
        :param current_action: (Optional) shape (b,1, act_dim)
        :param current_feedback: (Optional) shape (b,1, 1)
        :param attention_mask: (Optional) shape (b, t)
        """
        device = contexts.device

        batch_size, tar_horizon, _ = contexts.shape
        if tar_horizon > 0:
            last_feedback = feedbacks[:, -1, :].view(batch_size, 1, 1)
            feedbacks = feedbacks[:, :-1, :]
            feedbacks = torch.concat(
                [
                    torch.zeros(
                        (batch_size, 1, 1), dtype=torch.float, device=device
                    ),
                    feedbacks,
                ],
                dim=1,
            )  # padding the beginning with 0
        else:
            last_feedback = torch.zeros(
                (batch_size, 1, 1), dtype=torch.float, device=device
            )

        # combine the t-1 feedback with the t observation

        contexts = torch.concat(
            [feedbacks, contexts],
            dim=2,
        )  # concat the feedback with the context
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones(
                (batch_size, tar_horizon), dtype=torch.float, device=device
            )

        embedded_obs = self.obs_embedding(contexts)

        if self.same_embedding:
            act_values = torch.concat(
                [
                    torch.zeros(
                        (
                            batch_size,
                            tar_horizon,
                            self.obs_dim + 1 - self.act_dim,
                        ),
                        dtype=torch.float,
                        device=device,
                    ),
                    act_values,
                ],
                dim=2,
            )
            embedded_act = self.obs_embedding(act_values)

        else:
            embedded_act = self.act_embedding(act_values)

        # Embed the current contexts/action/ (stop if None)
        if current_ctx != None:
            current_ctx = torch.concat(
                [last_feedback, current_ctx],
                dim=2,
            )
        if self.same_embedding and current_action != None:
            current_action = torch.concat(
                [
                    torch.zeros(
                        (
                            batch_size,
                            tar_horizon,
                            self.obs_dim + 1 - self.act_dim,
                        ),
                        dtype=torch.float,
                        device=device,
                    ),
                    current_action,
                ],
                dim=2,
            )
        latest = [current_ctx, current_action]
        apply_cls = [self.obs_embedding, self.act_embedding]
        extra = []
        for inp, cls in zip(latest, apply_cls):
            if inp is not None:
                extra.append(cls(inp))  # (b,1, hidden_size)
            else:
                break
        num_extra = len(
            extra
        )  # number of non-empty current contexts/action/reward
        extra = (
            None if num_extra == 0 else torch.cat(extra, dim=1)
        )  # (b, i, hidden_size)

        # Stack the input
        input_seq = stack_seq(embedded_obs, embedded_act, extra)
        input_seq = self.layer_norm_in_embedding(input_seq)

        attention_mask = (
            attention_mask.unsqueeze(-1).repeat((1, 1, 2)).view(batch_size, -1)
        )  # (b, 2*t)

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
        pred_acts = output["last_hidden_state"][
            :, ::2, :
        ]  # (b, t, hidden_size) based on the tokens of observations+context
        pred_obs = output["last_hidden_state"][
            :, 1::2, :
        ]  # (b, t, hidden_size) based on the tokens of actions

        if self.output_attention:
            return (
                self.act_head(pred_acts),
                self.obs_head(pred_obs),
                output["attentions"],
            )
        elif self.output_hidden_states:
            return (
                self.act_head(pred_acts),
                self.obs_head(pred_obs),
                output.hidden_states,
            )
        else:
            return self.act_head(pred_acts), self.obs_head(pred_obs)

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
        """
        :param contexts: (b, t, ctx_dim)
        :param acts: (b, t, act_dim)
        :param feedbacks: (b, t, 1)
        :param current_ctx: (Optional) (b, obs_dim)
        """

        if (
            self.tar_length is not None
        ):  # truncate to tar length, where care the recent tar_length steps
            contexts = contexts[:, -self.tar_length :, :]
            acts = acts[:, -self.tar_length :, :]
            feedbacks = feedbacks[:, -self.tar_length :, :]
            batch_size = contexts.shape[0]

            # pad all tokens to sequence length All tokens are padded at the beginning of the sequence
            attention_mask = torch.cat(
                [
                    torch.zeros(
                        (batch_size, self.tar_length - contexts.shape[1])
                    ),
                    torch.ones((batch_size, contexts.shape[1])),
                ],
                dim=1,
            )
            attention_mask = attention_mask.to(
                dtype=torch.long, device=contexts.device
            ).reshape(batch_size, -1)
            contexts = torch.cat(
                [
                    torch.zeros(
                        (
                            contexts.shape[0],
                            self.tar_length - contexts.shape[1],
                            self.obs_dim,
                        ),
                        device=contexts.device,
                    ),
                    contexts,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            acts = torch.cat(
                [
                    torch.zeros(
                        (
                            acts.shape[0],
                            self.tar_length - acts.shape[1],
                            self.act_dim,
                        ),
                        device=acts.device,
                    ),
                    acts,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            feedbacks = torch.cat(
                [
                    torch.zeros(
                        (
                            feedbacks.shape[0],
                            self.tar_length - feedbacks.shape[1],
                            1,
                        ),
                        device=feedbacks.device,
                    ),
                    feedbacks,
                ],
                dim=1,
            ).to(dtype=torch.float32)
        else:
            attention_mask = None

        if self.action_type == "discrete":  # discrete action case
            results = self.forward(
                contexts,
                None,
                acts,
                feedbacks,
                current_ctx,
                None,
                None,
                None,
                attention_mask,
                None,
                None,
            )
            return_preds = results[1]
            action_preds = results[0]
            # probs (batch_size, output_act_dim)
            probs = (
                torch.nn.functional.softmax(
                    action_preds[:, -1, :] / temp, dim=-1
                )
                .cpu()
                .detach()
                .numpy()
            )
            sample_action = [
                np.random.choice(self.output_act_dim, p=prob) for prob in probs
            ]
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
                contexts,
                None,
                acts,
                feedbacks,
                current_ctx,
                None,
                None,
                None,
                attention_mask,
                None,
                None,
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


def stack_seq(contexts, acts, extra=None) -> torch.Tensor:
    """
    :param contexts: shape (b, t, hidden_size)
    :param act: shape (b, t, hidden_size)
    :param extra: (Optional) shape (b, i, hidden_size) where i can be 1, 2
    :return: shape (b, 3*t+i, hidden_size)
    """
    batch_size, tar_horizon, _ = contexts.shape
    stacked = (
        torch.stack((contexts, acts), dim=1)
        .permute(0, 2, 1, 3)
        .reshape(batch_size, 2 * tar_horizon, -1)
    )
    if extra is None:
        return stacked
    else:
        return torch.concat([stacked, extra], dim=1)

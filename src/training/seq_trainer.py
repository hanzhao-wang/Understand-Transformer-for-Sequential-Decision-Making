import torch

from training.trainer import Trainer
from training.utils import get_batch_single


class SequenceTrainer(Trainer):

    def train_step(self):
        if self.curr != None and self.iter_num <= self.curr["end_iter"]:
            len_unit = self.curr["unit_inc"]
            pool_size = int(self.target_length / len_unit)
            target_length = int(
                (
                    ((self.iter_num - 1) // self.curr["repeat_len"]) % pool_size
                    + 1
                )
                * len_unit
            )
        else:
            target_length = self.target_length

        (
            obs,
            action_ids,
            action_values,
            rewards,
            step_ids,
            attention_mask,
            target_as,
        ) = get_batch_single(
            self.obs_list,
            self.act_id_list,
            self.act_value_list,
            self.rew_list,
            self.opt_action_list,
            target_length,
            self.batch_size,
            self.state_dim,
            self.act_value_dim,
            self.device,
        )

        action_preds, rew_preds = self.model.forward(
            obs,
            action_ids,
            action_values,
            rewards,
            None,
            None,
            None,
            None,
            attention_mask,
            step_ids,
            None,
        )
        """
        if self.act_output_dim==1: #continuous action
            action_preds=action_value_preds
        else:
            action_preds = action_id_preds
        """

        action_preds = action_preds.reshape(-1, self.act_output_dim)[
            attention_mask.reshape(-1) > 0
        ]  # reshape to (batch_size*seq_len, action_dim) and remove padding
        if self.act_output_dim == 1:
            action_preds = action_preds.reshape(-1)

        if self.act_value_dim == 1 or self.task_name == "MAB":
            action_target = target_as.reshape(-1)[
                attention_mask.reshape(-1) > 0
            ]
        else:
            action_target = target_as.reshape(-1, self.act_value_dim)[
                attention_mask.reshape(-1) > 0
            ]

        rew_preds = rew_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        rew_target = rewards.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        loss = self.loss_fn(
            None,
            action_preds,
            rew_preds,
            None,
            action_target,
            rew_target,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        with torch.no_grad():
            if (
                self.act_output_dim == self.act_value_dim
                and self.task_name != "MAB"
            ):
                self.diagnostics["training/action_error"] = (
                    torch.mean((action_preds - action_target) ** 2)
                    .detach()
                    .cpu()
                    .item()
                )
            else:
                self.diagnostics["training/action_error"] = (
                    torch.mean(
                        (
                            torch.argmax(action_preds, dim=-1).reshape(-1)
                            - action_target.reshape(-1)
                        )
                        ** 2
                    )
                    .detach()
                    .cpu()
                    .item()
                )
            self.diagnostics["training/rew_error"] = (
                torch.mean((rew_preds - rew_target) ** 2).detach().cpu().item()
            )
            self.diagnostics["training/win_len"] = target_length

            # self.diagnostics['pred_action'].append(action_preds.detach().cpu().numpy())
        return loss.detach().cpu().item()

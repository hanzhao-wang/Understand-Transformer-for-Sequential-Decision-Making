import os
import time

import numpy as np
import torch

from training.utils import get_batch_single


class Trainer:

    def __init__(
        self,
        task_args,
        multigpu_args,
        model,
        target_len,
        optimizer,
        batch_size,
        loss_fn,
        scheduler=None,
        eval_fns=None,
        curr=0,
        FT_time=0,
        model_type="ICL",
    ):
        self.multigpu_args = multigpu_args
        self.device = (
            "cuda"
            if not multigpu_args.distributed
            else torch.device("cuda", multigpu_args.local_rank)
        )

        self.model = model  # model to be trained
        self.target_length = target_len  # length to truncate the trajectory to
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.loss_fn = loss_fn  # loss function to be used
        self.scheduler = scheduler  # scheduler to be used
        self.eval_fns = (
            [] if eval_fns is None else eval_fns
        )  # list of evaluation functions to monitor the training

        self.diagnostics = dict()
        self.start_time = time.time()
        self.state_dim = task_args.state_dim
        self.act_value_dim = task_args.act_value_dim
        self.act_output_dim = task_args.act_output_dim
        self.action_type = task_args.action_type
        self.task_name = task_args.task_name
        self.data_name = task_args.data_name
        self.exp_name = task_args.exp_name
        self.gen_method = task_args.gen_method
        self.fine_tune = task_args.fine_tune
        self.FT_time = FT_time
        self.model_type = model_type

        # load the data
        data_path = (
            "../../data/"
            + task_args.task_name
            + "/"
            + task_args.data_name
            + "/"
            + task_args.gen_method
        )

        # load csv files
        if task_args.fine_tune:
            self.obs_list = np.load(data_path + "/states_ft.npy")
            self.act_id_list = np.load(data_path + "/act_ids_ft.npy")
            self.act_value_list = np.load(data_path + "/act_values_ft.npy")
            self.rew_list = np.load(data_path + "/rewards_ft.npy")
            self.opt_action_list = np.load(data_path + "/opt_action_ft.npy")
            print("load data for fine-tune")
        else:
            self.obs_list = np.load(data_path + "/states.npy")
            self.act_id_list = np.load(data_path + "/act_ids.npy")
            self.act_value_list = np.load(data_path + "/act_values.npy")
            self.rew_list = np.load(data_path + "/rewards.npy")
            self.opt_action_list = np.load(data_path + "/opt_action.npy")

        self.obs_list_ori = np.load(data_path + "/states.npy")
        self.act_id_list_ori = np.load(data_path + "/act_ids.npy")
        self.act_value_list_ori = np.load(data_path + "/act_values.npy")
        self.rew_list_ori = np.load(data_path + "/rewards.npy")
        self.opt_action_list_ori = np.load(data_path + "/opt_action.npy")

        self.best_val_loss = float("inf")
        self.iter_num = 0
        self.curr = curr

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):
        self.iter_num = iter_num

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):

            train_loss = self.train_step()
            ########check the multi-GPU error if need#######
            # if self.multigpu_args.distributed:
            #    print(self.multigpu_args.local_rank,train_loss)
            ###############
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs["time/training"] = time.time() - train_start

        eval_start = time.time()
        if (not self.multigpu_args.distributed) or (
            self.multigpu_args.local_rank == 0
        ):
            self.model.eval()
            for eval_fn in self.eval_fns:
                outputs, data = eval_fn(self.model)
                for k, v in outputs.items():
                    logs[f"evaluation/{k}"] = v
            if (
                logs["evaluation/regret_mean_0"]
                + logs["evaluation/regret_std_0"]
                < self.best_val_loss
            ):
                self.best_val_loss = (
                    logs["evaluation/regret_mean_0"]
                    + logs["evaluation/regret_std_0"]
                )
                if not os.path.exists(
                    "../../model_logs/"
                    + self.task_name
                    + "/"
                    + self.exp_name
                    + "/"
                    + self.gen_method
                ):
                    os.makedirs(
                        "../../model_logs/"
                        + self.task_name
                        + "/"
                        + self.exp_name
                        + "/"
                        + self.gen_method
                    )
                model_name = "/best_model_" + self.model_type + ".pth"

                torch.save(
                    self.model.state_dict(),
                    "../../model_logs/"
                    + self.task_name
                    + "/"
                    + self.exp_name
                    + "/"
                    + self.gen_method
                    + model_name,
                )
            # save the model for every 10 iterations
            elif iter_num % 10 == 0:
                if not os.path.exists(
                    "../../model_logs/"
                    + self.task_name
                    + "/"
                    + self.exp_name
                    + "/"
                    + self.gen_method
                ):
                    os.makedirs(
                        "../../model_logs/"
                        + self.task_name
                        + "/"
                        + self.exp_name
                        + "/"
                        + self.gen_method
                    )

                model_name = (
                    "/model_" + self.model_type + str(iter_num) + ".pth"
                )
                torch.save(
                    self.model.state_dict(),
                    "../../model_logs/"
                    + self.task_name
                    + "/"
                    + self.exp_name
                    + "/"
                    + self.gen_method
                    + model_name,
                )
            if (self.FT_time != 0) and (iter_num > self.FT_time - 2):
                batch_size, length = (
                    np.array(data["states_list"]).squeeze().shape[0],
                    np.array(data["states_list"]).squeeze().shape[1],
                )

                self.obs_list = (
                    np.array(data["states_list"])
                    .squeeze()
                    .reshape(batch_size, length, -1)
                )
                self.act_id_list = np.array(data["act_ids_list"]).squeeze()
                self.act_value_list = np.array(
                    data["act_values_list"]
                ).squeeze()
                self.rew_list = np.array(data["rewards_list"]).squeeze()
                self.opt_action_list = np.array(data["opt_acts_list"]).squeeze()

                indices = np.random.choice(
                    int(self.obs_list_ori.shape[0] / 2),
                    int(self.obs_list.shape[0] / 2),
                    replace=False,
                )

                if self.action_type == "discrete":
                    self.opt_action_list = np.expand_dims(
                        self.opt_action_list, axis=2
                    )

                self.obs_list = np.concatenate(
                    (self.obs_list_ori[indices, :], self.obs_list), axis=0
                )
                self.act_id_list = np.concatenate(
                    (self.act_id_list_ori[indices, :], self.act_id_list), axis=0
                )
                self.act_value_list = np.concatenate(
                    (self.act_value_list_ori[indices, :], self.act_value_list),
                    axis=0,
                )
                self.rew_list = np.concatenate(
                    (self.rew_list_ori[indices, :], self.rew_list), axis=0
                )
                self.opt_action_list = np.concatenate(
                    (
                        self.opt_action_list_ori[indices, :],
                        self.opt_action_list,
                    ),
                    axis=0,
                )
        elif (self.FT_time != 0) and (iter_num > self.FT_time - 2):
            self.model.eval()
            for eval_fn in self.eval_fns:
                outputs, data = eval_fn(self.model)
            batch_size, length = (
                np.array(data["states_list"]).squeeze().shape[0],
                np.array(data["states_list"]).squeeze().shape[1],
            )

            self.obs_list = (
                np.array(data["states_list"])
                .squeeze()
                .reshape(batch_size, length, -1)
            )
            self.act_id_list = np.array(data["act_ids_list"]).squeeze()
            self.act_value_list = np.array(data["act_values_list"]).squeeze()
            self.rew_list = np.array(data["rewards_list"]).squeeze()
            self.opt_action_list = np.array(data["opt_acts_list"]).squeeze()
            if self.action_type == "discrete":
                self.opt_action_list = np.expand_dims(
                    self.opt_action_list, axis=2
                )
            indices = np.random.choice(
                int(self.obs_list_ori.shape[0] / 2),
                int(self.obs_list.shape[0] / 2),
                replace=False,
            )

            self.obs_list = np.concatenate(
                (self.obs_list_ori[indices, :], self.obs_list), axis=0
            )
            self.act_id_list = np.concatenate(
                (self.act_id_list_ori[indices, :], self.act_id_list), axis=0
            )
            self.act_value_list = np.concatenate(
                (self.act_value_list_ori[indices, :], self.act_value_list),
                axis=0,
            )
            self.rew_list = np.concatenate(
                (self.rew_list_ori[indices, :], self.rew_list), axis=0
            )
            self.opt_action_list = np.concatenate(
                (self.opt_action_list_ori[indices, :], self.opt_action_list),
                axis=0,
            )

        logs["time/total"] = (time.time() - self.start_time) / 60
        logs["time/evaluation"] = (time.time() - eval_start) / 60
        logs["time/training"] = (eval_start - self.start_time) / 60
        logs["training/train_loss_mean"] = np.mean(train_losses)
        logs["training/train_loss_std"] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print("=" * 80)
            print(f"Iteration {iter_num}")
            for k, v in logs.items():
                print(f"{k}: {v}")

        return logs

    def train_step(self):
        (
            states,
            action_ids,
            actions,
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
            self.target_length,
            self.batch_size,
            self.state_dim,
            self.act_value_dim,
            self.device,
        )
        state_target, action_target, reward_target = (
            torch.clone(states),
            torch.clone(actions),
            torch.clone(rewards),
        )

        state_preds, action_preds, reward_preds = self.model.forward(
            states,
            actions,
            rewards,
            masks=None,
            attention_mask=attention_mask,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds,
            action_preds,
            reward_preds,
            state_target[:, 1:],
            action_target,
            reward_target[:, 1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

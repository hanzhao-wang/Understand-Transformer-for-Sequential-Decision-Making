import random

import gurobipy as gp
import numpy as np
import torch
from scipy.linalg import sqrtm

from evaluation.evaluate_episodes import gen_env
from testing.GPT_test import temp_seed, test_model
from utils.tsne_visualize import plot_model_tsne


class list_class:
    def __init__(self):
        self.opt_reward_list = []
        self.demands_new = []
        self.act_values_new = []
        self.act_ids_new = []
        self.rewards_new = []
        self.regs_new = []
        self.opt_a_new = []

    def append_(
        self, demand, opt_reward, act_value, act_id, reward, reg, opt_a
    ):
        self.demands_new.append(demand)
        self.opt_reward_list.append(opt_reward)
        self.act_values_new.append(act_value)
        self.act_ids_new.append(act_id)
        self.rewards_new.append(reward)
        self.regs_new.append(reg)
        self.opt_a_new.append(opt_a)
        return self


class DP_test:
    def __init__(self, args):
        self.args = args
        seed = self.args.train.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def Bayes_cal(self, states, rewards, acts):
        states = states.detach().cpu().numpy().squeeze()
        demands = rewards.detach().cpu().numpy().squeeze()
        prices = acts.squeeze()
        dim_num = self.args.task.state_dim
        Z_T = np.concatenate((states, states * prices.reshape(-1, 1)), axis=1)
        task = self.args.task.params
        finite_envs_num = task["finite_envs_num"]
        need_square = task["need_square"]
        seed = task["seed"]
        err_std = task["err_std"]

        if self.args.task.task_name == "DP_ctx":
            arm_space = np.linspace(0.1, 5, self.args.task.act_output_dim)

        if finite_envs_num != 0:
            with temp_seed(seed):
                finite_envs_list = {
                    "beta": [
                        (np.random.rand(dim_num) * 2 + 1) / np.sqrt(dim_num)
                        for i in range(finite_envs_num)
                    ],
                    "gamma": [
                        -(np.random.rand(dim_num) + 0.1) / np.sqrt(dim_num)
                        for i in range(finite_envs_num)
                    ],
                }
                if need_square:
                    # sample True/False for each environment
                    finite_envs_list.update(
                        {
                            "square": [
                                True if i < (finite_envs_num / 2) else False
                                for i in range(finite_envs_num)
                            ]
                        }
                    )

        if need_square:
            envs_list = [
                (
                    finite_envs_list["beta"][i],
                    finite_envs_list["gamma"][i],
                    finite_envs_list["square"][i],
                )
                for i in range(finite_envs_num)
            ]
        else:
            envs_list = [
                (
                    finite_envs_list["beta"][i],
                    finite_envs_list["gamma"][i],
                    False,
                )
                for i in range(finite_envs_num)
            ]

        weights = np.ones((self.args.eval.test_horizon, len(envs_list)))
        opt_acts = np.ones((self.args.eval.test_horizon, len(envs_list)))

        for k in range(len(envs_list)):
            paras = np.concatenate((envs_list[k][0], envs_list[k][1])).reshape(
                -1, 1
            )
            square = envs_list[k][2]
            if not square:
                if self.args.task.task_name == "DP_ctx":
                    opt_acts[0, k] = arm_space[
                        np.argmax(
                            (
                                np.matmul(envs_list[k][0], states[0])
                                + np.matmul(envs_list[k][1], states[0])
                                * arm_space
                            )
                            * arm_space
                        ).squeeze()
                    ]
                else:
                    opt_acts[0, k] = np.clip(
                        -np.matmul(envs_list[k][0], states[0])
                        / (2 * np.matmul(envs_list[k][1], states[0])),
                        0,
                        task["action_up"],
                    )

                for t in range(self.args.eval.test_horizon - 1):
                    weights[t + 1, k] = np.exp(
                        np.sum(
                            (
                                demands[: t + 1]
                                - np.matmul(
                                    Z_T[: t + 1, :].squeeze(), paras
                                ).squeeze()
                            )
                            ** 2
                            / (-2 * err_std**2)
                        )
                    )
                    if self.args.task.task_name == "DP_ctx":
                        opt_acts[t + 1, k] = arm_space[
                            np.argmax(
                                (
                                    np.matmul(envs_list[k][0], states[t + 1])
                                    + np.matmul(envs_list[k][1], states[t + 1])
                                    * arm_space
                                )
                                * arm_space
                            ).squeeze()
                        ]
                    else:
                        opt_acts[t + 1, k] = np.clip(
                            -np.matmul(envs_list[k][0], states[t + 1])
                            / (2 * np.matmul(envs_list[k][1], states[t + 1])),
                            0,
                            task["action_up"],
                        )
            else:
                if self.args.task.task_name == "DP_ctx":
                    opt_acts[0, k] = arm_space[
                        np.argmax(
                            (
                                np.matmul(envs_list[k][0], states[0]) ** 2
                                - np.matmul(envs_list[k][1], states[0]) ** 2
                                * arm_space
                            )
                            * arm_space
                        ).squeeze()
                    ]
                else:
                    opt_acts[0, k] = np.clip(
                        np.matmul(envs_list[k][0], states[0]) ** 2
                        / (2 * np.matmul(envs_list[k][1], states[0]) ** 2),
                        0,
                        task["action_up"],
                    )

                for t in range(self.args.eval.test_horizon - 1):
                    values = (
                        np.matmul(states[: t + 1, :], envs_list[k][0]) ** 2
                        - np.matmul(states[: t + 1, :], envs_list[k][1]) ** 2
                        * prices[: t + 1]
                    )
                    weights[t + 1, k] = np.exp(
                        np.sum(
                            (demands[: t + 1] - values.squeeze()) ** 2
                            / (-2 * err_std**2)
                        )
                    )
                    if self.args.task.task_name == "DP_ctx":
                        opt_acts[t + 1, k] = arm_space[
                            np.argmax(
                                (
                                    np.matmul(envs_list[k][0], states[t + 1])
                                    ** 2
                                    - np.matmul(envs_list[k][1], states[t + 1])
                                    ** 2
                                    * arm_space
                                )
                                * arm_space
                            ).squeeze()
                        ]
                    else:
                        opt_acts[t + 1, k] = np.clip(
                            np.matmul(envs_list[k][0], states[t + 1]) ** 2
                            / (
                                2
                                * np.matmul(envs_list[k][1], states[t + 1]) ** 2
                            ),
                            0,
                            task["action_up"],
                        )
        weights = np.clip(weights, None, 1e6)
        weights = weights / np.sum(weights, axis=1).reshape(-1, 1)

        if self.args.task.task_name == "DP_ctx":
            bayes_acts = np.zeros((self.args.eval.test_horizon))
            for t in range(self.args.eval.test_horizon):
                bayes_acts[t] = opt_acts[t, np.argmax(weights[t])]
        else:
            bayes_acts = np.sum(opt_acts * weights, axis=1)
        return bayes_acts, weights

    def Bayes_check(self, val_num, test_len=None):
        test_len = self.args.eval.test_horizon

        self.args.eval.num_eval_episodes = 1
        Bayes_acts, GPT_acts, Opt_acts = [], [], []
        extras, states, rewards, acts, opt_actions, regs, cum_regs = test_model(
            self.args,
            test_env_cov=None,
            history_data=None,
            verbose=False,
            batch_size=val_num,
        )
        for i in range(val_num):

            Bayes_act, weights = self.Bayes_cal(states[i], rewards[i], acts[i])
            Bayes_acts.append(Bayes_act)
            # change one-hot actions to index
            acts_ = acts[i]
            GPT_acts.append(acts_)
            Opt_acts.append(opt_actions[i])

        return Bayes_acts, GPT_acts, Opt_acts

    def traj_compare(
        self,
        args,
        val_num,
        truncate_len,
        train_steps=[50, 70, 90, 110, 120],
        seed=123,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        max_ep_len = args.eval.test_horizon

        test_env_covs = []
        traj_dict = {}
        traj_dict["OrcTS_mean"] = {}
        traj_dict["OrcTS_mean"]["states"] = []
        traj_dict["OrcTS_mean"]["actions"] = []
        traj_dict["OrcTS_mean"]["rewards"] = []
        # traj_dict['Opt']={}
        # traj_dict['Opt']['states']=[]
        # traj_dict['Opt']['actions']=[]
        # traj_dict['Opt']['rewards']=[]

        for train_step in train_steps:
            traj_dict[str(train_step)] = {}
            for result in ["states", "actions", "rewards"]:
                traj_dict[str(train_step)][result] = []
        _, cov_fix, _ = gen_env(args.task, max_ep_len)
        for i in range(val_num):
            if self.args.task.params["need_square"]:
                square = True
                while square != False:
                    env, covariates, env_info = gen_env(args.task, max_ep_len)
                    test_env_cov = (env, cov_fix, env_info)
                    square = env.square
            else:
                env, covariates, env_info = gen_env(args.task, max_ep_len)
                test_env_cov = (env, cov_fix, env_info)
            act_ids_new, acts_new, _, regs_new, opt_actions, demands = (
                self.benchmark_test(
                    args, "OrcTS_mean", test_env_cov, None, None
                )
            )
            traj_dict["OrcTS_mean"]["states"].append(
                np.array(covariates).squeeze()[:truncate_len, :]
            )
            traj_dict["OrcTS_mean"]["actions"].append(
                np.array(acts_new).squeeze()[:truncate_len]
            )
            traj_dict["OrcTS_mean"]["rewards"].append(
                np.array(demands).squeeze()[:truncate_len]
            )
            # traj_dict['Opt']['actions'].append(np.array(opt_actions).squeeze()[:truncate_len])
            # traj_dict['Opt']['states'].append(np.array(covariates).squeeze()[:truncate_len,:])
            # traj_dict['Opt']['rewards'].append(np.array(demands).squeeze()[:truncate_len])

            test_env_covs.append(test_env_cov)
        traj_dict["OrcTS_mean"]["states"] = np.array(
            traj_dict["OrcTS_mean"]["states"]
        )
        traj_dict["OrcTS_mean"]["actions"] = np.array(
            traj_dict["OrcTS_mean"]["actions"]
        )
        traj_dict["OrcTS_mean"]["rewards"] = np.array(
            traj_dict["OrcTS_mean"]["rewards"]
        )
        # traj_dict['Opt']['states']=np.array(traj_dict['Opt']['states'])
        # traj_dict['Opt']['actions']=np.array(traj_dict['Opt']['actions'])
        # traj_dict['Opt']['rewards']=np.array(traj_dict['Opt']['rewards'])
        for train_step in train_steps:
            for setting in test_env_covs:
                setting[0].reset()
            model_name = "model_ICL" + str(train_step) + ".pth"
            extras, states, rewards, acts, opt_actions, regs, cum_regs = (
                test_model(
                    args,
                    test_env_covs,
                    None,
                    verbose=False,
                    batch_size=val_num,
                    model_name=model_name,
                )
            )
            traj_dict[str(train_step)]["states"] = [
                np.array(setting[1]).squeeze()[:truncate_len, :]
                for setting in test_env_covs
            ]
            traj_dict[str(train_step)]["states"] = np.array(
                traj_dict[str(train_step)]["states"]
            )
            traj_dict[str(train_step)]["actions"] = np.array(
                acts.squeeze()[:, :truncate_len]
            )
            traj_dict[str(train_step)]["rewards"] = (
                rewards.detach().cpu().numpy().squeeze()[:, :truncate_len]
            )
        plot_model_tsne(traj_dict)

    def compare(
        self,
        args,
        val_num,
        paras_tuning,
        method_names=["ILSE", "CILS", "TS", "OrcTS", "OrcTS_mean", "GPT"],
        seed=123,
        gen_method=None,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        args.eval.num_eval_episodes = 1
        max_ep_len = args.eval.test_horizon
        if paras_tuning:
            hyper_paras_dict = {}
            for method_name in method_names:
                if method_name in ["CILS", "TS"]:
                    hyper_paras_dict[method_name] = self.tuning(
                        args, method_name
                    )

        results_dict = {}

        if gen_method != None:
            test_env_covs = []
            for i in range(val_num):
                env, covariates, env_info = gen_env(args.task, max_ep_len)
                env.square = True if gen_method == "Square" else False
                test_env_cov = (env, covariates, env_info)
                test_env_covs.append(test_env_cov)
        else:
            test_env_covs = None

        for method_name in method_names:

            results_dict[method_name] = {}
            for result in ["acts", "opt_acts", "cum_regs"]:
                results_dict[method_name][result] = []

        for method_name in method_names:
            if paras_tuning and method_name in ["CILS", "TS"]:
                hyper_paras = hyper_paras_dict[method_name]
            else:
                hyper_paras = None

            history_data = None

            if method_name == "GPT":
                if gen_method != None:
                    for i in range(val_num):
                        test_env_covs[i][0].reset()

                if args.eval.output_probs == True:
                    (
                        probs,
                        states,
                        rewards,
                        acts,
                        opt_actions,
                        regs,
                        cum_regs,
                    ) = test_model(
                        args,
                        test_env_covs,
                        history_data,
                        verbose=False,
                        batch_size=val_num,
                    )
                    results_dict[method_name]["probs"].append(probs)
                else:
                    (
                        extras,
                        states,
                        rewards,
                        acts,
                        opt_actions,
                        regs,
                        cum_regs,
                    ) = test_model(
                        args,
                        test_env_covs,
                        history_data,
                        verbose=False,
                        batch_size=val_num,
                    )
                results_dict[method_name]["acts"] = acts
                results_dict[method_name]["cum_regs"] = cum_regs
                results_dict[method_name]["opt_acts"] = opt_actions
            else:
                for i in range(val_num):
                    if gen_method != None:
                        t_e_c = test_env_covs[i]
                        t_e_c[0].reset()
                    else:
                        t_e_c = None
                    (
                        act_ids_new,
                        acts_new,
                        rewards_new,
                        regs_new,
                        opt_actions,
                        _,
                    ) = self.benchmark_test(
                        args, method_name, t_e_c, history_data, hyper_paras
                    )
                    results_dict[method_name]["acts"].append(acts_new)
                    results_dict[method_name]["cum_regs"].append(
                        np.cumsum(regs_new)
                    )
                    results_dict[method_name]["opt_acts"].append(opt_actions)
        return results_dict

    def benchmark_test(
        self,
        args,
        method_name="ILSE",
        test_env_cov=None,
        history_data=None,
        hyper_paras=None,
    ):
        L = list_class()
        state_dim = args.task.state_dim
        max_ep_len = args.eval.test_horizon

        if test_env_cov == None:
            # sample the environment
            env, covariates, env_info = gen_env(args.task, max_ep_len)
        else:
            env, covariates, env_info = test_env_cov

        if history_data == None:
            episode_length = 0
            p_T = []
            sigma_t_inv = 10 * np.eye(state_dim * 2)
            sigma_t = 0.1 * np.eye(state_dim * 2)
            q_t = np.zeros((state_dim * 2))
            theta_hat = np.array([0.1] * (state_dim) + [-0.1] * (state_dim))

        else:
            episode_length = history_data["episode_length"]
            x_T = np.array(history_data["states"])
            p_T = np.array(history_data["act_values"]).squeeze()

            Z_T = np.concatenate((x_T, x_T * p_T.reshape(-1, 1)), axis=1)

            demands = np.array(history_data["rewards"])
            sigma_t = np.matmul(Z_T.T, Z_T) + 0.1 * np.eye(state_dim * 2)
            sigma_t_inv = np.linalg.inv(sigma_t)
            q_t = np.matmul(Z_T.T, demands)
            theta_hat = np.matmul(sigma_t_inv, q_t).squeeze()
            theta_hat_ori = theta_hat
            p_T = p_T.tolist()
        if method_name[:5] != "OrcTS":
            while episode_length < max_ep_len:

                cur_cov = covariates[env.time]
                a = np.matmul(theta_hat[:state_dim], cur_cov)
                b = np.matmul(theta_hat[state_dim:], cur_cov)

                if method_name == "ILSE":
                    para = [a, b]

                    if args.task.task_name == "DP_ctx":
                        reward_est = (
                            para[0] + para[1] * env.arm_space
                        ) * env.arm_space
                        sel_arm = np.argmax(reward_est)
                        p = env.arm_space[sel_arm]
                    else:
                        p = np.clip(-a / (2 * b), 0, env.action_ub)

                elif method_name == "CILS":
                    para = [a, b]

                    if args.task.task_name == "DP_ctx":
                        reward_est = (
                            para[0] + para[1] * env.arm_space
                        ) * env.arm_space
                        sel_arm = np.argmax(reward_est)
                        p = env.arm_space[sel_arm]
                    else:
                        p = np.clip(-a / (2 * b), 0, env.action_ub)
                    if hyper_paras == None:
                        kappa = 0.1
                    else:
                        kappa = hyper_paras
                    if episode_length > 0:
                        delta = p - np.mean(p_T)
                        if np.abs(delta) < kappa * pow(
                            episode_length + 1, -1 / 4
                        ):
                            sign = 1 if delta < 0 else -1
                            p = np.mean(p_T) + sign * kappa * pow(
                                episode_length + 1, -1 / 4
                            )
                            if args.task.task_name == "DP_ctx":
                                gap = env.arm_space - p
                                sel_arm = np.argmin(np.abs(gap))
                            else:
                                p = np.clip(-a / (2 * b), 0, env.action_ub)
                elif method_name == "TS":
                    if hyper_paras == None:
                        infla = (
                            np.sqrt(state_dim)
                            / 10
                            * np.sqrt(np.log(max_ep_len))
                        )
                    else:
                        infla = (
                            np.sqrt(state_dim)
                            * hyper_paras
                            * np.sqrt(np.log(max_ep_len))
                        )
                    X_tild = np.zeros((2, 2 * state_dim))
                    X_tild[0, :state_dim] = cur_cov
                    X_tild[1, state_dim:] = cur_cov
                    M_X = X_tild @ sigma_t_inv @ X_tild.T
                    sampling = np.random.normal(size=2)
                    para = np.array([a, b]) + infla * np.matmul(
                        sampling, sqrtm(M_X)
                    )

                    if args.task.task_name == "DP_ctx":
                        reward_est = (
                            para[0] + para[1] * env.arm_space
                        ) * env.arm_space
                        sel_arm = np.argmax(reward_est)
                    else:
                        p = np.clip(-para[0] / (2 * para[1]), 0, env.action_ub)
                if args.task.task_name == "DP_ctx":
                    (
                        demand,
                        action_ind,
                        reward,
                        done,
                        opt_arm_ind,
                        opt_reward,
                    ) = env.step(sel_arm, cur_cov, False)
                    action_value = env.arm_space[action_ind]
                else:
                    (
                        demand,
                        action_ind,
                        reward,
                        done,
                        opt_arm_ind,
                        opt_reward,
                    ) = env.step(p, cur_cov, False)
                    action_value = p
                p_T.append(action_value)

                z_t = np.concatenate((cur_cov, cur_cov * action_value))
                sigma_t = sigma_t + np.outer(z_t, z_t)
                mat_temp = np.matmul(sigma_t_inv, z_t)
                sigma_t_inv = sigma_t_inv - np.outer(mat_temp, mat_temp.T) / (
                    1 + np.matmul(z_t, mat_temp)
                )
                q_t = q_t + z_t * demand
                theta_hat = np.matmul(sigma_t_inv, q_t)

                opt_a = (
                    env.arm_space[opt_arm_ind]
                    if args.task.task_name == "DP_ctx"
                    else opt_arm_ind
                )
                L.append_(
                    demand,
                    opt_reward,
                    action_value,
                    action_ind,
                    reward,
                    opt_reward - reward,
                    opt_a,
                )

                episode_length += 1
        else:
            dim_num = self.args.task.state_dim
            task = self.args.task.params
            finite_envs_num = task["finite_envs_num"]
            need_square = task["need_square"]
            err_std = task["err_std"]
            seed = task["seed"]

            if self.args.task.task_name == "DP_ctx":
                arm_space = np.linspace(0.1, 5, self.args.task.act_output_dim)

            if finite_envs_num != 0:
                with temp_seed(seed):
                    finite_envs_list = {
                        "beta": [
                            (np.random.rand(dim_num) * 2 + 1) / np.sqrt(dim_num)
                            for i in range(finite_envs_num)
                        ],
                        "gamma": [
                            -(np.random.rand(dim_num) + 0.1) / np.sqrt(dim_num)
                            for i in range(finite_envs_num)
                        ],
                    }
                    if need_square:
                        # sample True/False for each environment
                        finite_envs_list.update(
                            {
                                "square": [
                                    True if i < (finite_envs_num / 2) else False
                                    for i in range(finite_envs_num)
                                ]
                            }
                        )

            if need_square:
                envs_list = [
                    (
                        finite_envs_list["beta"][i],
                        finite_envs_list["gamma"][i],
                        finite_envs_list["square"][i],
                    )
                    for i in range(finite_envs_num)
                ]
            else:
                envs_list = [
                    (
                        finite_envs_list["beta"][i],
                        finite_envs_list["gamma"][i],
                        False,
                    )
                    for i in range(finite_envs_num)
                ]

            weights = np.ones((len(envs_list)))
            opt_acts = np.ones((len(envs_list)))

            demands = []
            Z_T = []
            covs = []
            p_T = []

            while episode_length < max_ep_len:
                cur_cov = covariates[env.time]
                weights = np.clip(weights, None, 1e6)
                weights = weights / np.sum(weights)

                for k in range(len(envs_list)):
                    paras = np.concatenate(
                        (envs_list[k][0], envs_list[k][1])
                    ).reshape(-1, 1)
                    square = envs_list[k][2]
                    if not square:
                        if self.args.task.task_name == "DP_ctx":
                            opt_acts[k] = arm_space[
                                np.argmax(
                                    (
                                        np.matmul(envs_list[k][0], cur_cov)
                                        + np.matmul(envs_list[k][1], cur_cov)
                                        * arm_space
                                    )
                                    * arm_space
                                ).squeeze()
                            ]
                        else:
                            opt_acts[k] = np.clip(
                                -np.matmul(envs_list[k][0], cur_cov)
                                / (2 * np.matmul(envs_list[k][1], cur_cov)),
                                0,
                                task["action_up"],
                            )
                    else:
                        if self.args.task.task_name == "DP_ctx":
                            opt_acts[k] = arm_space[
                                np.argmax(
                                    (
                                        np.matmul(envs_list[k][0], cur_cov) ** 2
                                        - np.matmul(envs_list[k][1], cur_cov)
                                        ** 2
                                        * arm_space
                                    )
                                    * arm_space
                                ).squeeze()
                            ]
                        else:
                            opt_acts[k] = np.clip(
                                np.matmul(envs_list[k][0], cur_cov) ** 2
                                / (
                                    2 * np.matmul(envs_list[k][1], cur_cov) ** 2
                                ),
                                0,
                                task["action_up"],
                            )

                if method_name == "OrcTS":
                    act = opt_acts[np.random.choice(len(envs_list), p=weights)]

                elif method_name == "OrcTS_mean":
                    if self.args.task.task_name != "DP_ctx":
                        act = np.sum(opt_acts * weights)
                    else:
                        env_samp = np.argmax(weights)
                        act = opt_acts[env_samp]

                if args.task.task_name == "DP_ctx":
                    (
                        demand,
                        action_ind,
                        reward,
                        done,
                        opt_arm_ind,
                        opt_reward,
                    ) = env.step(act, cur_cov, False)
                    action_value = env.arm_space[action_ind]
                else:
                    (
                        demand,
                        action_ind,
                        reward,
                        done,
                        opt_arm_ind,
                        opt_reward,
                    ) = env.step(act, cur_cov, False)
                    action_value = act

                p_T.append(action_value)
                demands.append(demand)
                covs.append(cur_cov)

                z_T = np.concatenate((cur_cov, cur_cov * action_value))
                Z_T.append(z_T)

                for k in range(len(envs_list)):
                    paras = np.concatenate(
                        (envs_list[k][0], envs_list[k][1])
                    ).reshape(-1, 1)
                    square = envs_list[k][2]
                    if not square:
                        weights[k] = np.exp(
                            np.sum(
                                (
                                    np.array(demands)[: episode_length + 1]
                                    - np.matmul(
                                        np.array(Z_T)[
                                            : episode_length + 1, :
                                        ].squeeze(),
                                        paras,
                                    ).squeeze()
                                )
                                ** 2
                                / (-2 * err_std**2)
                            )
                        )
                    else:
                        values = np.matmul(
                            np.array(covs)[: episode_length + 1, :],
                            envs_list[k][0],
                        ) ** 2 - np.matmul(
                            np.array(covs)[: episode_length + 1, :],
                            envs_list[k][1],
                        ) ** 2 * np.array(
                            p_T
                        )
                        weights[k] = np.exp(
                            np.sum(
                                (
                                    np.array(demands)[: episode_length + 1]
                                    - values.squeeze()
                                )
                                ** 2
                                / (-2 * err_std**2)
                            )
                        )

                opt_a = (
                    env.arm_space[opt_arm_ind]
                    if args.task.task_name == "DP_ctx"
                    else opt_arm_ind
                )
                L.append_(
                    demand,
                    opt_reward,
                    action_value,
                    action_ind,
                    reward,
                    opt_reward - reward,
                    opt_a,
                )

                episode_length += 1

        return (
            L.act_ids_new,
            L.act_values_new,
            L.rewards_new,
            L.regs_new,
            L.opt_a_new,
            L.demands_new,
        )

    def tuning(self, args, method_name="CILS"):
        regs_mean_list = []
        if method_name == "CILS":
            Hpara_list = np.linspace(0.01, 1 + 0.01, 20)
        elif method_name == "TS":
            Hpara_list = np.linspace(0.01, 1 + 0.01, 20)
        for hyper_paras in Hpara_list:
            regs_list = []
            for i in range(20):
                (
                    act_ids_new,
                    act_values_new,
                    rewards_new,
                    regs_new,
                    opt_a_new,
                    _,
                ) = self.benchmark_test(
                    args,
                    method_name,
                    test_env_cov=None,
                    history_data=None,
                    hyper_paras=hyper_paras,
                )
                regs_list.append(np.sum(regs_new))
            regs_mean_list.append(np.mean(regs_list))
        best_hyper = Hpara_list[np.argmin(regs_mean_list)]
        return best_hyper

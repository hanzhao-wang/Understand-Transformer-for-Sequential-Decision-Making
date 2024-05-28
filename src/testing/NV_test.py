import math
import random

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
from scipy import stats

from evaluation.evaluate_episodes import gen_env
from testing.GPT_test import temp_seed, test_model


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


def z_hat_estimate(demands, h_b_ratio, arm_space):
    demands = np.array(demands)
    cost_list = []
    for arm in arm_space:
        cost = 1 * (demands - arm) * (demands > arm) + h_b_ratio * (
            arm - demands
        ) * (demands <= arm)
        cost_list.append(np.mean(cost))
    z_hat = arm_space[np.argmin(cost_list)]

    return z_hat


def quantile_regression(demands, h_b_ratio, covs, new_cov):
    quantile = 1 / (1 + h_b_ratio)
    data = pd.DataFrame({"demand": demands})
    dim = covs.shape[-1]
    covs = covs.reshape(-1, dim)
    for i in range(dim):
        data["X" + str(i)] = covs[:, i]
    model = smf.quantreg(
        "demand ~ " + "+".join(["X" + str(i) for i in range(dim)]), data
    )
    res = model.fit(q=quantile, max_iter=100)
    X_new = pd.DataFrame({"X" + str(i): [new_cov[i]] for i in range(dim)})
    pred = res.predict(X_new)[0]
    return pred


class NV_test:
    def __init__(self, args):
        self.args = args
        seed = self.args.train.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def Bayes_cal(self, states, rewards, acts, median=True):
        h_b_ratio = states.detach().cpu().numpy().squeeze()[0][1]
        demands = rewards.detach().cpu().numpy().squeeze()
        dim_num = self.args.task.state_dim
        cov_dim = dim_num - 2

        if cov_dim > 1:
            covs = states.detach().cpu().numpy().squeeze()[:, 2:]

        task = self.args.task.params
        finite_envs_num = task["finite_envs_num"]
        seed = task["seed"]
        need_square = task["need_square"]

        task = self.args.task.params

        if self.args.task.task_name == "NV":
            arm_num = self.args.task.act_output_dim
            arm_space = np.linspace(0, arm_num - 1, arm_num)

        if finite_envs_num != 0:
            with temp_seed(seed):
                finite_envs_list = {
                    "beta": [
                        np.random.uniform(low=0, high=3, size=cov_dim)
                        for i in range(finite_envs_num)
                    ],
                    "demand_para": [
                        np.random.uniform(low=1, high=10, size=1)[0]
                        for i in range(finite_envs_num)
                    ],
                }
                if need_square:
                    finite_envs_list.update(
                        {
                            "square": [
                                True if i < (finite_envs_num / 2) else False
                                for i in range(finite_envs_num)
                            ]
                        }
                    )

        weights = np.ones((self.args.eval.test_horizon, finite_envs_num))
        opt_acts = np.ones((self.args.eval.test_horizon, finite_envs_num))

        for k in range(finite_envs_num):
            para = finite_envs_list["demand_para"][k]
            beta = finite_envs_list["beta"][k]
            square = finite_envs_list["square"][k] if need_square else False
            if cov_dim > 1:
                if square:
                    coeffs = (covs @ beta.squeeze()) ** 2
                else:
                    coeffs = covs @ beta.squeeze()
            else:
                coeffs = np.zeros(self.args.eval.test_horizon)
            if self.args.task.task_name == "NV":
                opt_acts[:, k] = (
                    np.ceil(stats.randint.ppf(1 / (1 + h_b_ratio), 0, para + 1))
                    + coeffs
                )
                opt_acts[:, k] = np.clip(opt_acts[:, k], 0, arm_num - 1)
            else:
                opt_acts[:, k] = (
                    stats.uniform.ppf(1 / (1 + h_b_ratio), 0, para) + coeffs
                )
                opt_acts[:, k] = np.clip(opt_acts[:, k], 0, task["action_up"])
            for t in range(self.args.eval.test_horizon - 1):
                lower_flag = demands[: t + 1] - coeffs[: t + 1] <= para
                upper_flag = demands[: t + 1] > coeffs[: t + 1]
                if np.all(lower_flag) and np.all(upper_flag):
                    weights[t + 1, k] = 1 / para ** (t + 1)
                else:
                    weights[t + 1, k] = 0
                if self.args.task.task_name == "NV":
                    opt_acts[t + 1, k] = np.clip(
                        opt_acts[t + 1, k]
                        - states.detach().cpu().numpy().squeeze()[t + 1, 0],
                        0,
                        arm_num - 1,
                    )
                else:
                    opt_acts[t + 1, k] == np.clip(
                        opt_acts[t + 1, k]
                        - states.detach().cpu().numpy().squeeze()[t + 1, 0],
                        0,
                        task["action_up"],
                    )

        weights = weights / np.sum(weights, axis=1).reshape(-1, 1)

        if median:
            bayes_acts = np.zeros(self.args.eval.test_horizon)
            for t in range(self.args.eval.test_horizon):
                opt_acts_sort_idx = np.argsort(opt_acts[t])
                weights_sort = weights[t][opt_acts_sort_idx]
                cum_weights = np.cumsum(weights_sort)
                idx = np.argmax(cum_weights > 0.5)
                bayes_acts[t] = opt_acts[t][opt_acts_sort_idx][idx]
        else:  # mean
            bayes_acts = np.sum(opt_acts * weights, axis=1)
        return bayes_acts, weights

    def Bayes_check(self, val_num, median=True):
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

            Bayes_act, weights = self.Bayes_cal(
                states[i], rewards[i], acts[i], median
            )
            Bayes_acts.append(Bayes_act)
            # change one-hot actions to index
            acts_ = acts[i]
            GPT_acts.append(acts_)
            Opt_acts.append(opt_actions[i])
        return Bayes_acts, GPT_acts, Opt_acts

    def compare(
        self,
        args,
        val_num,
        paras_tuning,
        method_names=["ERM", "OGD", "OrcTS", "OrcTS_mean", "GPT"],
        seed=123,
        gen_method=None,
    ):
        seed = args.train.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        max_ep_len = args.eval.test_horizon

        args.eval.num_eval_episodes = 1

        if paras_tuning:
            hyper_paras_dict = {}
            for method_name in method_names:
                if method_name in ["OGD"]:
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
            if paras_tuning and method_name in ["OGD"]:
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
                    _, states, rewards, acts, opt_actions, regs, cum_regs = (
                        test_model(
                            args,
                            test_env_covs,
                            history_data,
                            verbose=False,
                            batch_size=val_num,
                        )
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
                        demands_new,
                        act_ids_new,
                        acts_new,
                        rewards_new,
                        regs_new,
                        opt_actions,
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
        method_name="GPT",
        test_env_cov=None,
        history_data=None,
        hyper_paras=None,
    ):

        state_dim = args.task.state_dim - 2
        max_ep_len = args.eval.test_horizon
        act_output_dim = args.task.act_output_dim
        L = list_class()

        if test_env_cov == None:
            # sample the environment
            env, covariates, env_info = gen_env(args.task, max_ep_len)
        else:
            env, covariates, env_info = test_env_cov

        if history_data == None:
            episode_length = 0
            demand_T = []
            z_hat = (
                random.randint(0, act_output_dim - 1)
                if act_output_dim > 1
                else np.random.uniform(0, env.action_ub)
            )
            h_b_ratio = env.h_b_ratio
            arm_space = (
                env.arms_space
                if act_output_dim > 1
                else np.linspace(0, env.action_ub, 100)
            )
            if method_name == "OGD":

                if hyper_paras == None:
                    step_size = 1
                else:
                    step_size = hyper_paras

                epsilon = (
                    2
                    / (math.sqrt(episode_length + 1) * max(1, h_b_ratio))
                    * step_size
                )
                if covariates.shape[-1] == 1:
                    z_hat = (
                        random.randint(0, act_output_dim - 1)
                        if act_output_dim > 1
                        else np.random.uniform(0, env.action_ub)
                    )
                else:
                    z_hat = np.random.uniform(
                        low=0, high=3, size=covariates.shape[-1]
                    )
        else:
            episode_length = history_data["episode_length"]
            demand_T = np.array(history_data["rewards"])
            demand_T = demand_T.tolist()
            h_b_ratio = env.h_b_ratio
            arm_space = env.arms_space
            z_hat = z_hat_estimate(demand_T, h_b_ratio, arm_space)

            if method_name == "OGD":
                episode_length = 0
                max_ep_len -= history_data["episode_length"]

                if hyper_paras == None:
                    step_size = 1
                else:
                    step_size = hyper_paras
                demand_larger_flag = (
                    history_data["rewards"][-1]
                    == history_data["act_values"][-1]
                )

                epsilon = (
                    2
                    / (math.sqrt(episode_length + 1) * max(1, h_b_ratio))
                    * step_size
                )
                if covariates.shape[-1] == 1:
                    z_hat = (
                        random.randint(0, act_output_dim - 1)
                        - epsilon * h_b_ratio
                        if demand_larger_flag == False
                        else random.randint(0, act_output_dim - 1) + epsilon
                    )
                else:
                    z_hat = np.random.uniform(
                        low=0, high=3, size=covariates.shape[-1]
                    )
        if method_name[:5] != "OrcTS":
            while episode_length < max_ep_len:
                if method_name == "ERM":
                    if episode_length > 0:
                        tar_stock_level = quantile_regression(
                            demand_T,
                            h_b_ratio,
                            covariates[:episode_length],
                            covariates[episode_length],
                        )
                    else:
                        tar_stock_level = (
                            np.random.uniform(0, act_output_dim - 1)
                            if act_output_dim > 1
                            else np.random.uniform(0, env.action_ub)
                        )
                    (
                        demand_larger_flag,
                        sale,
                        tar_order_level,
                        opt_order,
                        reward,
                        reg,
                        done,
                        opt_reward,
                    ) = env.step(
                        tar_stock_level, covariates[episode_length], False
                    )
                    action_value = tar_order_level
                    action_idx = (
                        np.argmin(np.abs(arm_space - tar_order_level))
                        if args.task.task_name == "NV"
                        else action_value
                    )
                    demand_T.append(sale)

                elif method_name == "OGD":
                    tar_stock_level = (
                        z_hat @ covariates[episode_length]
                        if covariates.shape[-1] > 1
                        else z_hat
                    )
                    tar_stock_level = (
                        np.ceil(tar_stock_level)
                        if np.random.uniform()
                        < (tar_stock_level - np.floor(tar_stock_level))
                        else np.floor(tar_stock_level)
                    )
                    if args.task.task_name == "NV":
                        gaps = arm_space - tar_stock_level + env.state
                        # if all gaps are negative, then choose the largest one
                        if np.all(gaps < 0):
                            tar_order_idx = np.argmax(gaps)
                        else:
                            gaps[gaps < 0] = np.inf
                            tar_order_idx = np.argmin(gaps)
                        tar_stock_level = arm_space[tar_order_idx] + env.state
                    else:
                        tar_stock_level = (
                            np.clip(
                                tar_stock_level - env.state, 0, env.action_ub
                            )
                            + env.state
                        )

                    (
                        demand_larger_flag,
                        sale,
                        tar_order_level,
                        opt_order,
                        reward,
                        reg,
                        done,
                        opt_reward,
                    ) = env.step(
                        tar_stock_level, covariates[episode_length], False
                    )
                    action_value = tar_order_level
                    action_idx = (
                        np.argmin(np.abs(arm_space - tar_order_level))
                        if args.task.task_name == "NV"
                        else 0
                    )

                    # OGD
                    if hyper_paras == None:
                        step_size = 1
                    else:
                        step_size = hyper_paras
                    epsilon = (
                        2
                        / (math.sqrt(episode_length + 1) * max(1, h_b_ratio))
                        * step_size
                    )
                    epsilon = (
                        epsilon
                        if covariates.shape[-1] == 1
                        else epsilon
                        * (1 - np.exp(-(episode_length + 1)))
                        * (1 / math.sqrt(episode_length + 1))
                    )  # shinkage

                    if covariates.shape[-1] == 1:

                        z_hat = (
                            z_hat - epsilon * h_b_ratio
                            if demand_larger_flag == False
                            else z_hat + epsilon
                        )
                    else:
                        z_hat = (
                            z_hat
                            - epsilon * h_b_ratio * covariates[episode_length]
                            if demand_larger_flag == False
                            else z_hat + epsilon * covariates[episode_length]
                        )
                    if args.task.task_name == "NV":
                        z_hat = np.clip(
                            z_hat, 0, act_output_dim - 1 + env.state
                        )
                    else:
                        z_hat = (
                            np.clip(z_hat, 0, env.action_ub + env.state)
                            if covariates.shape[-1] == 1
                            else np.clip(z_hat, 0, 40)
                        )

                opt_a = (
                    arm_space[opt_order]
                    if args.task.task_name == "NV"
                    else opt_order
                )
                L.append_(
                    sale,
                    opt_reward,
                    action_value,
                    action_idx,
                    reward,
                    reg,
                    opt_a,
                )
                episode_length += 1
        else:
            task = args.task.params
            h_b_ratio = env.h_b_ratio

            finite_envs_num = task["finite_envs_num"]
            seed = task["seed"]
            if finite_envs_num != 0:
                with temp_seed(seed):
                    finite_envs_list = {
                        "beta": [
                            np.random.uniform(low=0, high=3, size=state_dim)
                            for i in range(finite_envs_num)
                        ],
                        "demand_para": [
                            np.random.uniform(low=1, high=10, size=1)[0]
                            for i in range(finite_envs_num)
                        ],
                    }
                if task["need_square"]:
                    finite_envs_list.update(
                        {
                            "square": [
                                True if i < (finite_envs_num / 2) else False
                                for i in range(finite_envs_num)
                            ]
                        }
                    )
            weights = np.ones(finite_envs_num)
            opt_acts = np.ones(finite_envs_num)

            demands = []
            covs = []

            while episode_length < max_ep_len:
                cur_cov = covariates[env.time]
                weights = np.clip(weights, None, 1e6)
                weights = weights / np.sum(weights)

                for k in range(finite_envs_num):
                    para = finite_envs_list["demand_para"][k]
                    beta = finite_envs_list["beta"][k]
                    square = (
                        finite_envs_list["square"][k]
                        if task["need_square"]
                        else False
                    )
                    if covariates.shape[-1] > 1:
                        if square:
                            coeffs = (cur_cov @ beta) ** 2
                        else:
                            coeffs = cur_cov @ beta
                    else:
                        coeffs = 0
                    if self.args.task.task_name == "NV":
                        opt_acts[k] = (
                            np.ceil(
                                stats.randint.ppf(
                                    1 / (1 + h_b_ratio), 0, para + 1
                                )
                            )
                            + coeffs
                        )
                    else:
                        opt_acts[k] = (
                            stats.uniform.ppf(1 / (1 + h_b_ratio), 0, para)
                            + coeffs
                        )

                if method_name == "OrcTS":
                    act = opt_acts[np.random.choice(finite_envs_num, p=weights)]

                elif method_name == "OrcTS_mean":
                    if self.args.task.task_name != "NV":
                        act = np.sum(opt_acts * weights)
                    else:
                        env_samp = np.argmax(weights)
                        act = opt_acts[env_samp]

                tar_stock_level = act
                if args.task.task_name == "NV":
                    gaps = arm_space - tar_stock_level + env.state
                    # if all gaps are negative, then choose the largest one
                    if np.all(gaps < 0):
                        tar_order_idx = np.argmax(gaps)
                    else:
                        gaps[gaps < 0] = np.inf
                        tar_order_idx = np.argmin(gaps)
                    tar_stock_level = arm_space[tar_order_idx] + env.state
                else:
                    tar_stock_level = (
                        np.clip(tar_stock_level - env.state, 0, env.action_ub)
                        + env.state
                    )

                (
                    demand_larger_flag,
                    sale,
                    tar_order_level,
                    opt_order,
                    reward,
                    reg,
                    done,
                    opt_reward,
                ) = env.step(tar_stock_level, covariates[episode_length], False)

                demands.append(sale)
                covs.append(cur_cov)
                for k in range(finite_envs_num):
                    para = finite_envs_list["demand_para"][k]
                    beta = finite_envs_list["beta"][k]
                    square = (
                        finite_envs_list["square"][k]
                        if task["need_square"]
                        else False
                    )
                    if covariates.shape[-1] > 1:
                        if square:
                            coeffs = (np.array(covs) @ beta) ** 2
                        else:
                            coeffs = np.array(covs) @ beta
                    else:
                        coeffs = 0
                    lower_flag = np.array(demands) - np.array(coeffs) <= para
                    upper_flag = np.array(demands) > np.array(coeffs)
                    if np.all(lower_flag) and np.all(upper_flag):
                        weights[k] = 1 / para ** (episode_length + 1)
                    else:
                        weights[k] = 0

                opt_a = (
                    arm_space[opt_order]
                    if args.task.task_name == "NV"
                    else opt_order
                )
                L.append_(
                    sale,
                    opt_reward,
                    tar_order_level,
                    tar_order_level,
                    reward,
                    reg,
                    opt_a,
                )
                episode_length += 1

        return (
            L.demands_new,
            L.act_ids_new,
            L.act_values_new,
            L.rewards_new,
            L.regs_new,
            L.opt_a_new,
        )

    def tuning(self, args, method_name="OGD"):
        regs_mean_list = []
        if method_name == "OGD":
            Hpara_list = np.linspace(0.1, 2 + 0.01, 20)
        for hyper_paras in Hpara_list:
            regs_list = []
            for i in range(20):
                (
                    demands_new,
                    act_ids_new,
                    act_values_new,
                    rewards_new,
                    regs_new,
                    opt_a_new,
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

import math
import random

import numpy as np
import torch
from scipy.linalg import cholesky

from envs.Bandits import MAB_env
from evaluation.evaluate_episodes import gen_env
from testing.GPT_test import temp_seed, test_model


class list_class:
    def __init__(self):
        self.opt_reward_list = []
        self.act_values_new = []
        self.act_ids_new = []
        self.rewards_new = []
        self.regs_new = []
        self.opt_a_new = []

    def append_(self, opt_reward, act_value, act_id, reward, reg, opt_a):
        self.opt_reward_list.append(opt_reward)
        self.act_values_new.append(act_value)
        self.act_ids_new.append(act_id)
        self.rewards_new.append(reward)
        self.regs_new.append(reg)
        self.opt_a_new.append(opt_a)
        return self


def sample_from_d_sphere(d):

    v = np.random.normal(0, 1, d)
    norm = np.linalg.norm(v)
    unit_vector = v / norm

    return unit_vector


class MAB_test:
    def __init__(self, args):
        self.args = args
        seed = self.args.train.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.act_num = args.task.act_output_dim
        self.dim_num = self.act_num

    def Bayes_cal(self, rewards, acts):

        rewards = rewards.detach().cpu().numpy().squeeze()
        Z_T = acts
        task = self.args.task.params
        finite_envs_num = task["finite_envs_num"]
        if finite_envs_num != 0:
            with temp_seed(task["seed"]):
                finite_envs_list = {
                    "theta": [
                        np.random.normal(0, 1, self.act_num)
                        for i in range(finite_envs_num)
                    ]
                }
        err_std = task["err_std"]

        weights = np.ones((self.args.eval.test_horizon, finite_envs_num))
        opt_acts = np.ones(finite_envs_num)
        bayes_acts = np.ones(self.args.eval.test_horizon)

        for k in range(finite_envs_num):
            paras = finite_envs_list["theta"][k]

            opt_acts[k] = np.argmax(paras)
            for t in range(self.args.eval.test_horizon - 1):
                weights[t + 1, k] = np.exp(
                    np.sum(
                        (
                            rewards[: t + 1]
                            - np.matmul(
                                Z_T[: t + 1, :].squeeze(), paras
                            ).squeeze()
                        )
                        ** 2
                        / (-2 * err_std**2)
                    )
                )
        weights = np.clip(weights, None, 1e6)
        weights = weights / np.sum(weights, axis=1).reshape(-1, 1)
        for t in range(self.args.eval.test_horizon):
            bayes_acts[t] = opt_acts[np.argmax(weights[t])]
        return bayes_acts, weights

    def Bayes_check(self, val_num, seed=123):
        test_len = self.args.eval.test_horizon
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        args = self.args
        args.eval.temp = 0.001
        task = self.args.task.params
        finite_envs_num = task["finite_envs_num"]
        max_ep_len = args.eval.test_horizon
        if finite_envs_num != 0:
            with temp_seed(task["seed"]):
                finite_envs_list = {
                    "theta": [
                        np.random.normal(0, 1, self.act_num)
                        for i in range(finite_envs_num)
                    ]
                }
        err_std = task["err_std"]

        test_env_covs = []
        thetas = []
        for i in range(val_num):
            idx = np.random.choice(finite_envs_num)
            theta = finite_envs_list["theta"][idx]
            env = MAB_env(theta, horizon=max_ep_len, err_std=task["err_std"])
            test_env_covs.append((env, None, None))
            thetas.append(theta)

        self.args.eval.num_eval_episodes = 1
        Bayes_acts, GPT_acts, Opt_acts = [], [], []
        extras, states, rewards, acts, opt_actions, regs, cum_regs = test_model(
            args,
            test_env_covs,
            history_data=None,
            verbose=False,
            batch_size=val_num,
        )
        for i in range(val_num):
            Bayes_act, weights = self.Bayes_cal(rewards[i], acts[i])
            Bayes_act_ = [
                thetas[i][int(Bayes_act[j])] for j in range(max_ep_len)
            ]
            Bayes_acts.append(Bayes_act_)
            # change one-hot actions to index
            acts_ = acts[i].squeeze()
            acts_ = np.argmax(acts_, axis=1)
            acts_ = [thetas[i][int(acts_[j])] for j in range(max_ep_len)]
            GPT_acts.append(acts_)
            opt_acts_ = np.argmax(opt_actions[i], axis=1)
            opt_acts_ = [
                thetas[i][int(opt_acts_[j])] for j in range(max_ep_len)
            ]
            Opt_acts.append(opt_acts_)

        return Bayes_acts, GPT_acts, Opt_acts

    def benchmark_test(
        self,
        args,
        method_name="TS",
        test_env_cov=None,
        history_data=None,
        hyper_paras=None,
    ):
        L = list_class()
        max_ep_len = args.eval.test_horizon
        arms_num = self.act_num

        if test_env_cov == None:
            # sample the environment
            env, covariates, env_info = gen_env(args.task, max_ep_len)

        if history_data == None:
            episode_length = 0
            arms = []

        if method_name == "TS":
            if hyper_paras == None:
                infla = 1
            else:
                infla = hyper_paras
            mu_s = np.zeros(self.act_num)
            sel_s = np.zeros(self.act_num)
            while episode_length < max_ep_len:

                theta = [
                    np.random.normal(
                        mu_s[i],
                        infla
                        * math.sqrt(2 * np.log(max_ep_len) / (sel_s[i] + 1.0)),
                    )
                    for i in range(arms_num)
                ]
                action_ind = np.argmax(theta)

                act_idx, reward, done, opt_arm_ind, opt_reward = env.step(
                    action_ind
                )
                mu_s[action_ind] = (
                    mu_s[action_ind] * sel_s[action_ind] + reward
                ) / (sel_s[action_ind] + 1)
                sel_s[action_ind] += 1
                L.append_(
                    opt_reward,
                    act_idx,
                    act_idx,
                    reward,
                    opt_reward - reward,
                    opt_arm_ind,
                )
                episode_length += 1

        elif method_name == "UCB":
            if hyper_paras == None:
                infla = 1
            else:
                infla = hyper_paras
            counts = np.zeros(self.act_num)
            r_sums = np.zeros(self.act_num)
            while episode_length < max_ep_len:
                UCB = r_sums / np.clip(counts, 1, None) + infla * np.sqrt(
                    2 * np.log(max_ep_len) / np.clip(counts, 1, None)
                )
                act_idx = np.argmax(UCB)
                act_idx, reward, done, opt_arm_ind, opt_reward = env.step(
                    act_idx
                )
                counts[act_idx] += 1
                r_sums[act_idx] += reward

                L.append_(
                    opt_reward,
                    act_idx,
                    act_idx,
                    reward,
                    opt_reward - reward,
                    opt_arm_ind,
                )
                episode_length += 1
        # if method name start with OrcTS
        elif method_name[:5] == "OrcTS":

            rewards = []
            Z_T = []
            task = self.args.task.params
            if task["finite_envs_num"] != 0:
                with temp_seed(task["seed"]):
                    finite_envs_list = {
                        "theta": [
                            np.random.normal(0, 1, self.act_num)
                            for i in range(task["finite_envs_num"])
                        ]
                    }
            err_std = task["err_std"]
            weights = np.ones((task["finite_envs_num"]))
            opt_acts = np.ones((task["finite_envs_num"]))
            for k in range(task["finite_envs_num"]):
                paras = finite_envs_list["theta"][k]
                opt_acts[k] = np.argmax(paras)
            while episode_length < max_ep_len:

                weights = np.clip(weights, None, 1e6)
                weights = weights / np.sum(weights)

                if method_name == "OrcTS":
                    env_samp = np.random.choice(
                        task["finite_envs_num"], p=weights
                    )

                elif method_name == "OrcTS_argmax":
                    env_samp = np.argmax(weights)
                act_idx = opt_acts[env_samp]

                act_idx, reward, done, opt_arm_ind, opt_reward = env.step(
                    int(act_idx)
                )
                rewards.append(reward)
                Z_T.append(env.arm_space[act_idx])
                for k in range(task["finite_envs_num"]):
                    weights[k] = np.exp(
                        np.sum(
                            (
                                np.array(rewards)
                                - np.matmul(
                                    np.array(Z_T).squeeze(),
                                    np.array(finite_envs_list["theta"][k]),
                                )
                            )
                            ** 2
                            / (-2 * err_std**2)
                        )
                    )

                L.append_(
                    opt_reward,
                    act_idx,
                    act_idx,
                    reward,
                    opt_reward - reward,
                    opt_arm_ind,
                )
                episode_length += 1

        return (
            L.act_ids_new,
            L.act_values_new,
            L.rewards_new,
            L.regs_new,
            L.opt_a_new,
        )

    def tuning(self, args, method_name="TS"):
        regs_mean_list = []
        if method_name in ["TS", "UCB"]:
            Hpara_list = np.linspace(0.01, 1 + 0.01, 20)
        for hyper_paras in Hpara_list:
            regs_list = []
            for i in range(20):
                (
                    dact_ids_new,
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

    def compare(
        self,
        args,
        val_num,
        paras_tuning,
        method_names=["TS", "UCB", "OrcTS", "OrcTS_argmax", "GPT"],
        seed=123,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        args.eval.num_eval_episodes = 1

        dim_num = args.task.state_dim
        horizon = args.eval.test_horizon
        if paras_tuning:
            hyper_paras_dict = {}
            for method_name in method_names:
                if method_name in ["TS", "UCB"]:
                    hyper_paras_dict[method_name] = self.tuning(
                        args, method_name
                    )

        results_dict = {}
        for method_name in method_names:
            results_dict[method_name] = {}
            for result in ["acts", "opt_acts", "cum_regs"]:
                results_dict[method_name][result] = []

        for method_name in method_names:
            if paras_tuning and method_name in ["TS", "UCB"]:
                hyper_paras = hyper_paras_dict[method_name]
            else:
                hyper_paras = None

            history_data = None
            test_env_cov = None
            if method_name == "GPT":
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
                        test_env_cov,
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
                        test_env_cov,
                        history_data,
                        verbose=False,
                        batch_size=val_num,
                    )
                results_dict[method_name]["acts"] = 0  # np.argmax(acts,axis=2)
                results_dict[method_name]["cum_regs"] = cum_regs
                cum_regs_ = np.concatenate(
                    [np.zeros((val_num, 1)), cum_regs], axis=1
                )
                regs = cum_regs_[:, 1:] - cum_regs_[:, :-1]
                results_dict[method_name][
                    "opt_acts"
                ] = regs  # np.argmax(opt_actions,axis=2)
            else:
                for i in range(val_num):
                    (
                        act_ids_new,
                        acts_new,
                        rewards_new,
                        regs_new,
                        opt_actions,
                    ) = self.benchmark_test(
                        args,
                        method_name,
                        test_env_cov,
                        history_data,
                        hyper_paras,
                    )
                    results_dict[method_name]["acts"].append(np.zeros(horizon))
                    results_dict[method_name]["cum_regs"].append(
                        np.cumsum(regs_new)
                    )
                    results_dict[method_name]["opt_acts"].append(regs_new)
        return results_dict


class LinB_test(MAB_test):
    def __init__(self, args):
        self.args = args
        seed = self.args.train.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.dim_num = args.task.act_output_dim

    def Bayes_cal(self, rewards, acts):

        rewards = rewards.detach().cpu().numpy().squeeze()
        Z_T = acts
        task = self.args.task.params
        if task["finite_envs_num"] != 0:
            with temp_seed(task["seed"]):
                finite_envs_list = {
                    "theta": [
                        sample_from_d_sphere(self.dim_num)
                        for i in range(task["finite_envs_num"])
                    ]
                }
        err_std = task["err_std"]
        envs_list = finite_envs_list["theta"]

        weights = np.ones((self.args.eval.test_horizon, len(envs_list)))
        opt_acts = np.ones(
            (self.args.eval.test_horizon, len(envs_list), self.dim_num)
        )
        bayes_acts = np.ones((self.args.eval.test_horizon, self.dim_num))

        for k in range(len(envs_list)):
            paras = finite_envs_list["theta"][k]
            opt_acts[0, k, :] = paras

            for t in range(self.args.eval.test_horizon - 1):
                weights[t + 1, k] = np.exp(
                    np.sum(
                        (
                            rewards[: t + 1]
                            - np.matmul(
                                Z_T[: t + 1, :].squeeze(), paras
                            ).squeeze()
                        )
                        ** 2
                        / (-2 * err_std**2)
                    )
                )
                opt_acts[t + 1, k, :] = paras

        weights = np.clip(weights, None, 1e6)
        weights = weights / np.sum(weights, axis=1).reshape(-1, 1)
        # unsqueeze the weights to the same shape as opt_acts
        weights = np.expand_dims(weights, axis=2)
        bayes_acts = np.sum(weights * opt_acts, axis=1)
        return bayes_acts, weights

    def Bayes_check(self, val_num, seed=123):
        test_len = self.args.eval.test_horizon
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

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

            Bayes_act, weights = self.Bayes_cal(rewards[i], acts[i])
            Bayes_acts.append(Bayes_act)
            acts_ = acts[i]
            GPT_acts.append(acts_)
            Opt_acts.append(opt_actions[i])

        return Bayes_acts, GPT_acts, Opt_acts

    def benchmark_test(
        self,
        args,
        method_name="TS",
        test_env_cov=None,
        history_data=None,
        hyper_paras=None,
    ):
        L = list_class()
        max_ep_len = args.eval.test_horizon

        if test_env_cov == None:
            # sample the environment
            env, covariates, env_info = gen_env(args.task, max_ep_len)

        if history_data == None:
            episode_length = 0
            arms = []

        if method_name == "TS":
            if hyper_paras == None:
                infla = 1
            else:
                infla = hyper_paras
            actions = []
            rewards = []
            while episode_length < max_ep_len:
                if len(actions) != 0:

                    # compute the theta_hat and the covariance matrix
                    theta_hat = np.linalg.inv(
                        np.matmul(np.array(actions).T, np.array(actions))
                        + env.err_std * np.eye(self.dim_num)
                    ) @ np.matmul(np.array(actions).T, np.array(rewards))
                    cov_M = np.linalg.inv(
                        np.matmul(np.array(actions).T, np.array(actions))
                        + env.err_std * np.eye(self.dim_num)
                    )
                    # sample the theta from the posterior
                    theta_sample = np.random.multivariate_normal(
                        theta_hat, infla * 2 * np.log(max_ep_len) * cov_M
                    )
                    # get the best action
                    action = theta_sample / np.linalg.norm(theta_sample)
                else:
                    action = np.random.normal(0, 1, self.dim_num)

                act, reward, done, opt_arm_ind, opt_reward = env.step(action)
                actions.append(action)
                rewards.append(reward)
                L.append_(
                    opt_reward,
                    act,
                    act,
                    reward,
                    opt_reward - reward,
                    opt_arm_ind,
                )
                episode_length += 1

        elif method_name == "UCB":
            if hyper_paras == None:
                infla = 1
            else:
                infla = hyper_paras
            actions = []
            rewards = []
            while episode_length < max_ep_len:
                if len(actions) != 0:
                    theta_hat = np.linalg.inv(
                        np.matmul(np.array(actions).T, np.array(actions))
                        + env.err_std * np.eye(self.dim_num)
                    ) @ np.matmul(np.array(actions).T, np.array(rewards))
                    cov_M = np.linalg.inv(
                        np.matmul(np.array(actions).T, np.array(actions))
                        + env.err_std * np.eye(self.dim_num)
                    )
                    # compute the UCB by Monte Carlo sampling
                    UCB_pool = self.sample(
                        cov_M, theta_hat, 200, infla * 2 * np.log(max_ep_len)
                    )
                    # compute the norm of each column of the UCB_pool
                    UCB_norm = np.linalg.norm(UCB_pool, axis=0)
                    action = UCB_pool[:, np.argmax(UCB_norm)]
                else:
                    action = np.random.normal(0, 1, self.dim_num)
                action = action / np.linalg.norm(action)
                actions.append(action)

                act, reward, done, opt_arm_ind, opt_reward = env.step(action)
                rewards.append(reward)
                L.append_(
                    opt_reward,
                    act,
                    act,
                    reward,
                    opt_reward - reward,
                    opt_arm_ind,
                )
                episode_length += 1
        # if method name start with OrcTS
        elif method_name[:5] == "OrcTS":

            task = self.args.task.params
            if task["finite_envs_num"] != 0:
                with temp_seed(task["seed"]):
                    finite_envs_list = {
                        "theta": [
                            sample_from_d_sphere(self.dim_num)
                            for i in range(task["finite_envs_num"])
                        ]
                    }
            err_std = task["err_std"]

            weights = np.ones(len(finite_envs_list["theta"]))
            opt_acts = np.ones((len(finite_envs_list["theta"]), self.dim_num))
            for k in range(len(finite_envs_list["theta"])):
                opt_acts[k, :] = finite_envs_list["theta"][k]

            rewards = []
            Z_T = []

            while episode_length < max_ep_len:

                weights = np.clip(weights, None, 1e6)
                weights = weights / np.sum(weights)

                if method_name == "OrcTS":
                    env_samp = np.random.choice(
                        len(finite_envs_list["theta"]), p=weights
                    )
                    action = opt_acts[env_samp]

                elif method_name == "OrcTS_mean":

                    action = np.sum(
                        opt_acts * np.expand_dims(weights, axis=1), axis=0
                    )
                    action = action / np.linalg.norm(action)

                act_value, reward, done, opt_arm_ind, opt_reward = env.step(
                    action
                )
                rewards.append(reward)
                Z_T.append(action)
                for k in range(len(finite_envs_list["theta"])):
                    weights[k] = np.exp(
                        np.sum(
                            (
                                np.array(rewards)
                                - np.matmul(
                                    np.array(Z_T).squeeze(),
                                    np.array(finite_envs_list["theta"][k]),
                                )
                            )
                            ** 2
                            / (-2 * err_std**2)
                        )
                    )

                L.append_(
                    opt_reward,
                    act_value,
                    act_value,
                    reward,
                    opt_reward - reward,
                    opt_arm_ind,
                )
                episode_length += 1

        return (
            L.act_ids_new,
            L.act_values_new,
            L.rewards_new,
            L.regs_new,
            L.opt_a_new,
        )

    def compare(
        self,
        args,
        val_num,
        paras_tuning,
        method_names=["TS", "UCB", "OrcTS", "OrcTS_mean", "GPT"],
        seed=123,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        args.eval.num_eval_episodes = 1

        dim_num = args.task.state_dim
        horizon = args.eval.test_horizon
        if paras_tuning:
            hyper_paras_dict = {}
            for method_name in method_names:
                if method_name in ["TS", "UCB"]:
                    hyper_paras_dict[method_name] = self.tuning(
                        args, method_name
                    )

        results_dict = {}
        for method_name in method_names:
            results_dict[method_name] = {}
            for result in ["acts", "opt_acts", "cum_regs"]:
                results_dict[method_name][result] = []
        for method_name in method_names:

            if paras_tuning and method_name in ["TS", "UCB"]:
                hyper_paras = hyper_paras_dict[method_name]
            else:
                hyper_paras = None

            history_data = None
            test_env_cov = None
            if method_name == "GPT":
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
                        test_env_cov,
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
                        test_env_cov,
                        history_data,
                        verbose=False,
                        batch_size=val_num,
                    )
                results_dict[method_name]["acts"] = acts
                results_dict[method_name]["cum_regs"] = cum_regs
                results_dict[method_name]["opt_acts"] = opt_actions
            else:
                for i in range(val_num):
                    (
                        act_ids_new,
                        acts_new,
                        rewards_new,
                        regs_new,
                        opt_actions,
                    ) = self.benchmark_test(
                        args,
                        method_name,
                        test_env_cov,
                        history_data,
                        hyper_paras,
                    )
                    results_dict[method_name]["acts"].append(acts_new)
                    results_dict[method_name]["cum_regs"].append(
                        np.cumsum(regs_new)
                    )
                    results_dict[method_name]["opt_acts"].append(opt_actions)
        return results_dict

    def sample(self, S, z_hat, m_FA, Gamma_Threshold=1.0):
        nz = S.shape[0]
        z_hat = z_hat.reshape(nz, 1)

        X_Cnz = np.random.normal(size=(nz, m_FA))

        rss_array = np.sqrt(np.sum(np.square(X_Cnz), axis=0))
        kron_prod = np.kron(np.ones((nz, 1)), rss_array)

        X_Cnz = (
            X_Cnz / kron_prod
        )  # Points uniformly distributed on hypersphere surface

        R = np.ones((nz, 1)) * (np.power(np.random.rand(1, m_FA), (1.0 / nz)))

        unif_sph = R * X_Cnz
        # m_FA points within the hypersphere
        T = np.asmatrix(cholesky(S))  # Cholesky factorization of S => S=T’T

        unif_ell = T.H * unif_sph
        # Hypersphere to hyperellipsoid mapping

        # Translation and scaling about the center
        z_fa = unif_ell * np.sqrt(Gamma_Threshold) + (
            z_hat * np.ones((1, m_FA))
        )

        return np.array(z_fa)

import contextlib
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from envs import DPctx_env_cts, LinB_env, MAB_env, NV_env_cts


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def sample_from_d_sphere(d):

    v = np.random.normal(0, 1, d)
    norm = np.linalg.norm(v)
    unit_vector = v / norm

    return unit_vector


def gen_state(task_name, env, covariates, episode_length):
    if task_name in ["DP_ctx", "DP_ctx_cts"]:
        state__ = covariates[episode_length]
    elif task_name in ["NV", "NV_cts"]:
        state__ = [env.state, env.h_b_ratio] + covariates[
            episode_length
        ].tolist()
    elif task_name == "RM":
        # state includes the current inventory level, the current demand type's A and c, A_list and c_list for all demand types
        dtype = int(env.demand_types[env.time])
        state__ = (
            env.state.reshape(-1).tolist()
            + env.A_list[:, dtype].tolist()
            + [env.c_list[dtype]]
            + env.A_list.reshape(-1).tolist()
            + env.c_list.tolist()
        )
    elif task_name == "single_Q":
        state__ = [env.state, env.service_cost]
    elif task_name in ["MAB", "LinB"]:
        state__ = [0]
    return state__


def gen_env(task_args, max_ep_len):
    task_name = task_args.task_name
    task = task_args.params
    env_info = {}
    covariates = None
    if task_name == "MAB":
        arm_num = task_args.act_output_dim
        finite_envs_num = task["finite_envs_num"]
        if finite_envs_num == 0:
            theta = np.random.normal(0, 1, arm_num)
        else:
            with temp_seed(task["seed"]):
                finite_envs_list = {
                    "theta": [
                        np.random.normal(0, 1, arm_num)
                        for i in range(finite_envs_num)
                    ]
                }
            idx = np.random.choice(finite_envs_num)
            theta = finite_envs_list["theta"][idx]
        env = MAB_env(theta, horizon=max_ep_len, err_std=task["err_std"])
    elif task_name == "LinB":
        act_dim = task_args.act_output_dim
        finite_envs_num = task["finite_envs_num"]
        if finite_envs_num == 0:
            theta = sample_from_d_sphere(act_dim)
        else:
            with temp_seed(task["seed"]):
                finite_envs_list = {
                    "theta": [
                        sample_from_d_sphere(act_dim)
                        for i in range(finite_envs_num)
                    ]
                }
            idx = np.random.choice(finite_envs_num)
            theta = finite_envs_list["theta"][idx]
        env = LinB_env(theta, horizon=max_ep_len, err_std=task["err_std"])
    elif task_name == "DP_ctx_cts":
        dim_num = task_args.state_dim
        price_up = task["action_up"]
        err_std = task["err_std"]
        finite_envs_num = task["finite_envs_num"]
        seed = task["seed"]

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
                if task["need_square"]:
                    # sample True/False for each environment
                    finite_envs_list.update(
                        {
                            "square": [
                                True if i < (finite_envs_num / 2) else False
                                for i in range(finite_envs_num)
                            ]
                        }
                    )
            beta_space = finite_envs_list["beta"]
            length = len(beta_space)
            idx = np.random.choice(length)
            beta_star = beta_space[idx]
            gamma_space = finite_envs_list["gamma"]
            gamma_star = gamma_space[idx]
            if task["need_square"]:
                square_space = finite_envs_list["square"]
                square = square_space[idx]
            else:
                square = False
        else:
            beta_star = (np.random.rand(dim_num) * 2 + 1) / np.sqrt(dim_num)
            gamma_star = -(np.random.rand(dim_num) + 0.1) / np.sqrt(dim_num)
            if task["need_square"]:
                square = np.random.choice([True, False])
            else:
                square = False
        covariates = np.random.uniform(
            low=0, high=5, size=(max_ep_len, dim_num)
        ) / np.sqrt(dim_num)

        covariates[:, 0] = 1
        demand_func = [beta_star, gamma_star]

        env_info = {"beta_star": beta_star, "gamma_star": gamma_star}

        env = DPctx_env_cts(
            price_up,
            demand_func,
            horizon=max_ep_len,
            err_std=err_std,
            square=square,
        )
    elif task_name == "NV_cts":

        dim_num = (
            task_args.state_dim - 2
        )  # the first dimension is the inventory and the second dimension is the cost ratio
        finite_envs_num = task["finite_envs_num"]
        seed = task["seed"]

        covariates = np.random.uniform(
            low=0, high=3, size=(max_ep_len, dim_num)
        )  # only 1 or 2 to keep integer
        covariates[:, 0] = 1

        if finite_envs_num == 0:
            beta_star = np.random.uniform(low=0, high=3, size=dim_num)
            if dim_num == 1:
                beta_star = [0]
            h_b_ratio = np.random.uniform(low=0.5, high=2, size=1)[0]
            demand_para = np.random.uniform(low=1, high=10, size=1)[0]
            if task["need_square"]:
                square = np.random.choice([True, False])
            else:
                square = False

        else:
            with temp_seed(seed):
                finite_envs_list = {
                    "beta": [
                        np.random.uniform(low=0, high=3, size=dim_num)
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
            idx = np.random.choice(finite_envs_num)
            beta_star = finite_envs_list["beta"][idx]
            if dim_num == 1:
                beta_star = [0]
            h_b_ratio = np.random.uniform(low=0.5, high=2, size=1)[0]
            demand_para = finite_envs_list["demand_para"][idx]
            if task["need_square"]:
                square = finite_envs_list["square"][idx]
            else:
                square = False

        demand_func = {"name": "uniform", "para": [0, demand_para]}

        env = NV_env_cts(
            task["action_up"],
            demand_func=demand_func,
            horizon=max_ep_len,
            h_b_ratio=h_b_ratio,
            perishable=task["perishable"],
            coeff=beta_star,
            censor=task["censor"],
            square=square,
        )
        env_info = {}
    else:
        raise NotImplementedError
    return env, covariates, env_info


def gen_update_info(env, action_idx, task_name, covariates):
    if task_name == "MAB":
        action_ind, reward, done, opt_arm_ind, opt_reward = env.step(action_idx)
        action = env.arm_space[action_idx]
        opt_action = env.arm_space[opt_arm_ind]
        opt_act_ids = opt_arm_ind
        episode_reg = env.opt_reward - reward
        cur_obs = reward
    elif task_name == "LinB":

        action_ind, reward, done, opt_arm_ind, opt_reward = env.step(action_idx)
        action = action_idx
        opt_action = opt_arm_ind
        opt_act_ids = opt_arm_ind
        episode_reg = env.opt_reward - reward
        cur_obs = reward
    elif task_name == "NV_cts":
        tar_stock_level = (
            action_idx + env.state
        )  # here the action_idx is just the action
        _, demand, action, opt_order_action, reward, reg, done, opt_reward = (
            env.step(tar_stock_level, covariates[env.time], False)
        )  # use sale as 'demand', tar_order_level as 'action' here
        episode_reg = reg
        opt_arm_ind = opt_order_action  # now the target 'id' is just the value
        opt_action = opt_order_action
        opt_act_ids = opt_arm_ind
        cur_obs = demand
    elif task_name == "DP_ctx_cts":
        demand, action_ind, reward, done, opt_arm_ind, opt_reward = env.step(
            action_idx, covariates[env.time]
        )
        action = action_ind  # now the target 'id' is just the value
        opt_action = opt_arm_ind
        opt_act_ids = opt_arm_ind
        episode_reg = env.opt_reward - reward
        cur_obs = demand
    else:
        raise NotImplementedError
    return (
        env,
        action,
        episode_reg,
        cur_obs,
        opt_action,
        opt_reward,
        opt_act_ids,
    )


def evaluate_episode(
    task_args,
    eval_args,
    model,
    testing_mode=False,
    history_data=None,
    test_env_cov=None,
    batch_size=64,
):
    device = "cuda"
    model.eval()

    model.output_attention = eval_args.output_attention
    model.output_hidden_states = eval_args.output_hidden_states
    model.need_probs = eval_args.output_probs
    warm_up_steps = eval_args.warm_up_steps
    warm_up_pure_random = eval_args.warm_up_pure_random
    scaling_reward = eval_args.scaling_reward
    max_ep_len = eval_args.test_horizon
    max_train_len = eval_args.train_horizon

    task_name = task_args.task_name
    state_dim = task_args.state_dim
    act_value_dim = task_args.act_value_dim
    act_output_dim = task_args.act_output_dim
    task_params = task_args.params

    if test_env_cov == None:
        # sample the environment
        envs = []
        covs = []
        gen_infos = []
        for i in range(batch_size):
            env, covariates, env_info = gen_env(task_args, max_ep_len)
            envs.append(env)
            covs.append(covariates)
            gen_infos.append(env_info)
    else:
        envs = []
        covs = []
        gen_infos = []
        for i in range(batch_size):
            env, covariates, env_info = test_env_cov[i]
            envs.append(env)
            covs.append(covariates)
            gen_infos.append(env_info)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    if history_data == None:
        states = torch.zeros(
            (batch_size, 0, state_dim), device=device, dtype=torch.float32
        )
        act_values = torch.zeros(
            (batch_size, 0, act_value_dim), device=device, dtype=torch.float32
        )
        act_ids = torch.zeros((batch_size, 0), device=device, dtype=torch.long)
        rewards = torch.zeros(
            (batch_size, 0, 1), device=device, dtype=torch.float32
        )
        timesteps = torch.zeros(
            (batch_size, 0), device=device, dtype=torch.long
        )
        episode_length = 0
    else:
        states = torch.tensor(
            history_data["states"], device=device, dtype=torch.float32
        )
        act_values = torch.tensor(
            history_data["act_values"], device=device, dtype=torch.float32
        )
        act_ids = torch.tensor(
            history_data["act_ids"], device=device, dtype=torch.long
        )
        rewards = torch.tensor(
            history_data["rewards"], device=device, dtype=torch.float32
        )
        timesteps = torch.tensor(
            history_data["timesteps"], device=device, dtype=torch.long
        )
        # add first dimension
        states = states.reshape(1, states.shape[0], state_dim)
        act_values = act_values.reshape(1, act_values.shape[0], act_value_dim)
        act_ids = act_ids.reshape(1, act_ids.shape[0])
        rewards = rewards.reshape(1, rewards.shape[0], 1)
        timesteps = timesteps.reshape(1, timesteps.shape[0])
        episode_length = history_data["episode_length"]

    episode_regs = np.zeros(batch_size)
    opt_actions = np.zeros((batch_size, max_ep_len, act_value_dim))

    if task_args.action_type == "discrete":
        opt_act_ids = np.zeros((batch_size, max_ep_len))
    else:
        opt_act_ids = np.zeros((batch_size, max_ep_len, act_value_dim))
    opt_rewards = np.zeros((batch_size, max_ep_len))
    extras = []
    cum_regs = []
    if eval_args.output_probs == True:
        probs = []
        shift_time = env.shift_time

    while episode_length < max_ep_len:
        cur_state = []
        for i in range(batch_size):
            env = envs[i]
            covariates = covs[i]
            env_info = gen_infos[i]
            cur_state.append(
                gen_state(task_name, env, covariates, episode_length)
            )
        current_step_id = (
            torch.ones(batch_size, 1, device=device, dtype=torch.long)
            * episode_length
        )
        if hasattr(model, "module"):
            output_ = model.module
        else:
            output_ = model
        # output_=model if testing_mode else model.module
        results = output_.get_action(
            states.to(dtype=torch.float32),
            act_ids.to(dtype=torch.long),
            act_values.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            torch.tensor(
                np.array(cur_state), device=device, dtype=torch.float32
            ).reshape(batch_size, 1, state_dim),
            current_step_id,
            eval_args.temp,
        )
        action, _return = results[0], results[1]

        cur_state = (
            torch.tensor(np.array(cur_state))
            .to(device=device)
            .reshape(batch_size, 1, state_dim)
        )
        states = torch.cat([states, cur_state], dim=1)
        action = action.detach().cpu().numpy().squeeze()
        envs_new = []
        act_values_new = []
        cur_obs_new = []

        for i in range(batch_size):
            env = envs[i]
            covariates = covs[i]
            env_info = gen_infos[i]
            action_temp = action[i]
            (
                env,
                action_value,
                reg,
                cur_obs,
                opt_action,
                opt_reward,
                opt_act_id,
            ) = gen_update_info(env, action_temp, task_name, covariates)
            episode_regs[i] += reg
            envs_new.append(env)
            act_values_new.append(action_value)
            cur_obs_new.append(cur_obs)
            opt_actions[i, episode_length] = opt_action
            opt_act_ids[i, episode_length] = opt_act_id
            opt_rewards[i, episode_length] = opt_reward

            if eval_args.output_hidden_states == True:
                if episode_length == max_ep_len - 1:
                    extras = results[2]
                    env_info["opt_idx"] = opt_act_id[-1]
                    env_info["cur_obs"] = cur_obs
            if eval_args.output_attention == True:
                if episode_length == max_ep_len - 1:
                    extras = results[2]
            if eval_args.output_probs == True:
                probs.append(results[2])

        envs = envs_new

        act_values = torch.cat(
            [
                act_values,
                torch.tensor(np.array(act_values_new))
                .to(device=device)
                .reshape(batch_size, 1, act_value_dim),
            ],
            dim=1,
        )
        if task_args.action_type == "continuous":  # continuous action
            act_ids = torch.cat(
                [
                    act_ids,
                    torch.zeros(
                        (batch_size, 1), device=device, dtype=torch.long
                    ),
                ],
                dim=1,
            )
        else:
            action_idx = action
            act_ids = torch.cat(
                [
                    act_ids,
                    torch.tensor(action_idx)
                    .to(device=device)
                    .reshape(batch_size, 1),
                ],
                dim=1,
            )
        rewards = torch.cat(
            [
                rewards,
                torch.tensor(cur_obs_new, device=device).reshape(
                    batch_size, 1, 1
                ),
            ],
            dim=1,
        )

        if episode_length < max_train_len:
            timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((batch_size, 1), device=device, dtype=torch.long)
                    * (episode_length),
                ],
                dim=1,
            )

        cum_regs.append(episode_regs.copy())
        episode_length += 1
    if (
        scaling_reward
    ):  # normalize the regrets by dividing the cumulative optimal rewards
        cum_regs = np.array(cum_regs).T / np.clip(
            np.array(opt_rewards).sum(axis=1, keepdims=True), 0.1, None
        )
    else:
        cum_regs = np.array(cum_regs).T

    actions_list = act_values.detach().cpu().numpy().squeeze()

    if eval_args.output_probs:
        extras = {"probs": probs, "shift_time": shift_time}

    if eval_args.for_demand_pred:
        return (
            act_values.detach().cpu().numpy().squeeze(),
            act_ids.detach().cpu().numpy().squeeze(),
            rewards.detach().cpu().numpy().squeeze(),
        )
    else:
        return (
            extras,
            states,
            act_ids,
            act_values,
            rewards,
            opt_act_ids.squeeze(),
            cum_regs,
            opt_actions.squeeze(),
            actions_list,
        )

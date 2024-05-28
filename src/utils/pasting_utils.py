import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch

from evaluation.evaluate_episodes import gen_env, gen_state, gen_update_info


def pasting(
    task,
    model,
    test_env_cov=None,
    output_attention=False,
    output_hidden_states=False,
    cur_inv_changed=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device=device)
    model.output_attention = output_attention
    model.output_hidden_states = output_hidden_states
    task_name = task["name"]
    state_dim = task["state_dim"]
    act_value_dim = task["act_value_dim"]
    max_ep_len = task["test_horizon"]
    max_train_len = task["train_horizon"]
    scaling_reward = task["scaling_reward"]
    opt_reward_list = []
    if test_env_cov == None:
        # sample the environment
        env, covariates, env_info = gen_env(task, max_ep_len)
    else:
        env, covariates, env_info = test_env_cov

    states = torch.zeros((1, 0, state_dim), device=device, dtype=torch.float32)
    act_values = torch.zeros(
        (1, 0, act_value_dim), device=device, dtype=torch.float32
    )
    act_ids = torch.zeros((1, 0), device=device, dtype=torch.long)
    rewards = torch.zeros((1, 0, 1), device=device, dtype=torch.float32)
    timesteps = torch.zeros((1, 0), device=device, dtype=torch.long)
    episode_length = 0
    opt_action = []
    episode_reg = 0

    while episode_length < max_ep_len:

        if cur_inv_changed != None:
            if episode_length == cur_inv_changed["time"] - 1:
                changed_inv = (
                    cur_inv_changed["inv"]
                    if cur_inv_changed["inv"] != None
                    else env.state
                )
                if changed_inv == "random":
                    changed_inv = np.random.randint(low=0, high=10, size=1)[0]

                changed_hb = (
                    cur_inv_changed["h_b_ratio"]
                    if cur_inv_changed["h_b_ratio"] != None
                    else env.h_b_ratio
                )
                if changed_hb == "random":
                    changed_hb = np.random.uniform(low=0.5, high=2, size=1)[0]
                state_c = (
                    env.arms_space.tolist()
                    + [changed_inv, changed_hb]
                    + covariates[episode_length].tolist()
                )

                changed_action, changed_return = model.get_action(
                    states.to(dtype=torch.float32),
                    act_ids.to(dtype=torch.long),
                    act_values.to(dtype=torch.float32),
                    rewards.to(dtype=torch.float32),
                    timesteps.to(dtype=torch.long),
                    current_obs=torch.tensor(
                        state_c, device=device, dtype=torch.float32
                    ).reshape(1, 1, state_dim),
                    current_step_id=current_step_id,
                    temp=task["temprature"],
                )
                changed_opt_order_level = np.clip(
                    np.ceil(
                        stats.poisson.ppf(
                            env.critical_ratio, env.demand_func["para"]
                        )
                    )
                    - changed_inv,
                    0,
                    None,
                )
                changed_opt_order_level = np.clip(
                    changed_opt_order_level,
                    env.arms_space[0],
                    env.arms_space[-1],
                )

        cur_state = gen_state(task_name, env, covariates, episode_length)
        current_step_id = torch.tensor(
            episode_length, device=device, dtype=torch.long
        ).reshape(1, 1)

        results = model.get_action(
            states.to(dtype=torch.float32),
            act_ids.to(dtype=torch.long),
            act_values.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            current_obs=torch.tensor(
                cur_state, device=device, dtype=torch.float32
            ).reshape(1, 1, state_dim),
            current_step_id=current_step_id,
            temp=task["temprature"],
        )
        action, _return = results[0], results[1]

        cur_state = (
            torch.tensor(cur_state).to(device=device).reshape(1, 1, state_dim)
        )
        states = torch.cat([states, cur_state], dim=1)
        action = action.detach().cpu().numpy().squeeze()
        action_idx = action

        env, action_value, episode_reg, cur_obs, opt_action, opt_reward, _ = (
            gen_update_info(
                env, action_idx, task_name, covariates, opt_action, episode_reg
            )
        )

        act_values = torch.cat(
            [
                act_values,
                torch.tensor(action_value)
                .to(device=device)
                .reshape(1, 1, act_value_dim),
            ],
            dim=1,
        )
        act_ids = torch.cat(
            [act_ids, torch.tensor(action_idx).to(device=device).reshape(1, 1)],
            dim=1,
        )
        rewards = torch.cat(
            [rewards, torch.tensor(cur_obs, device=device).reshape(1, 1, 1)],
            dim=1,
        )
        if cur_inv_changed != None:
            if episode_length == cur_inv_changed["time"] - 1:
                return (
                    changed_action,
                    changed_opt_order_level,
                    action_value,
                    opt_action[-1],
                    state_c,
                    cur_state,
                )
        if episode_length < max_train_len:
            timesteps = torch.cat(
                [
                    timesteps,
                    torch.ones((1, 1), device=device, dtype=torch.long)
                    * (episode_length),
                ],
                dim=1,
            )

        opt_reward_list.append(opt_reward)

        episode_length += 1


def pasting_nv(variant, cur_inv_changed):
    seed = variant["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    task = variant["task"]

    model = torch.load(
        "logs/"
        + task["name"]
        + "/"
        + task["exp_name"]
        + "/"
        + task["gen_method"]
        + "/best_model.pth"
    )
    model.eval()
    c_a_list = []
    c_opt_list = []
    t_a_list = []
    t_opt_list = []
    c_state_list = []
    t_state_list = []
    with torch.no_grad():
        for i in range(variant["num_eval_episodes"]):
            (
                changed_action,
                changed_opt_order_level,
                action_value,
                opt_action,
                state_c,
                cur_state,
            ) = pasting(task, model, cur_inv_changed=cur_inv_changed)
            c_a_list.append(changed_action[0])
            c_opt_list.append(changed_opt_order_level)
            t_a_list.append(action_value)
            t_opt_list.append(opt_action)
            c_state_list.append(state_c)
            t_state_list.append(cur_state.squeeze().detach().cpu().numpy())
    # save the results
    c_a_list = np.array(c_a_list)
    c_opt_list = np.array(c_opt_list)
    t_a_list = np.array(t_a_list)
    t_opt_list = np.array(t_opt_list)
    c_state_list = np.array(c_state_list)
    t_state_list = np.array(t_state_list)
    save_path = "exp_results/NV_pasting_exp"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + "/pasting_nv.pkl", "wb") as f:
        pickle.dump(
            [
                c_a_list,
                c_opt_list,
                t_a_list,
                t_opt_list,
                c_state_list,
                t_state_list,
            ],
            f,
        )


def plot_nv(variant):
    data_ao_path = "exp_results/NV_pasting_exp/pasting_nv.pkl"
    with open(data_ao_path, "rb") as f:
        (
            c_a_list,
            c_opt_list,
            t_a_list,
            t_opt_list,
            c_state_list,
            t_state_list,
        ) = pickle.load(f)

    changed_a_changed_opt = c_a_list - c_opt_list
    unchanged_a_unchanged_opt = t_a_list - t_opt_list
    changed_a_unchanged_a = c_a_list - t_a_list
    changed_a_unchanged_opt = c_a_list - t_opt_list

    # plot to compare the difference of these three distirbutions, on a figure with 3 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    bins_num = 40
    # add the mean with steandard deviation on the plot as vertialcal line corresponding to each distribution
    axs[0, 0].hist(changed_a_changed_opt, bins=bins_num)
    axs[0, 0].axvline(
        changed_a_changed_opt.mean(), color="k", linestyle="dashed", linewidth=2
    )
    axs[0, 0].axvline(
        changed_a_changed_opt.mean() + changed_a_changed_opt.std(),
        color="k",
        linestyle="dashed",
        linewidth=1,
    )
    axs[0, 0].axvline(
        changed_a_changed_opt.mean() - changed_a_changed_opt.std(),
        color="k",
        linestyle="dashed",
        linewidth=1,
    )
    axs[0, 0].set_title("changed_a_changed_opt")
    axs[0, 1].hist(unchanged_a_unchanged_opt, bins=bins_num)
    axs[0, 1].axvline(
        unchanged_a_unchanged_opt.mean(),
        color="k",
        linestyle="dashed",
        linewidth=2,
    )
    axs[0, 1].axvline(
        unchanged_a_unchanged_opt.mean() + unchanged_a_unchanged_opt.std(),
        color="k",
        linestyle="dashed",
        linewidth=1,
    )
    axs[0, 1].axvline(
        unchanged_a_unchanged_opt.mean() - unchanged_a_unchanged_opt.std(),
        color="k",
        linestyle="dashed",
        linewidth=1,
    )
    axs[0, 1].set_title("unchanged_a_unchanged_opt")
    axs[1, 0].hist(changed_a_unchanged_a, bins=bins_num)
    axs[1, 0].axvline(
        changed_a_unchanged_a.mean(), color="k", linestyle="dashed", linewidth=2
    )
    axs[1, 0].axvline(
        changed_a_unchanged_a.mean() + changed_a_unchanged_a.std(),
        color="k",
        linestyle="dashed",
        linewidth=1,
    )
    axs[1, 0].axvline(
        changed_a_unchanged_a.mean() - changed_a_unchanged_a.std(),
        color="k",
        linestyle="dashed",
        linewidth=1,
    )
    axs[1, 0].set_title("changed_a_unchanged_a")
    axs[1, 1].hist(changed_a_unchanged_opt, bins=bins_num)
    axs[1, 1].axvline(
        changed_a_unchanged_opt.mean(),
        color="k",
        linestyle="dashed",
        linewidth=2,
    )
    axs[1, 1].axvline(
        changed_a_unchanged_opt.mean() + changed_a_unchanged_opt.std(),
        color="k",
        linestyle="dashed",
        linewidth=1,
    )
    axs[1, 1].axvline(
        changed_a_unchanged_opt.mean() - changed_a_unchanged_opt.std(),
        color="k",
        linestyle="dashed",
        linewidth=1,
    )
    axs[1, 1].set_title("changed_a_unchanged_opt")
    plt.show()

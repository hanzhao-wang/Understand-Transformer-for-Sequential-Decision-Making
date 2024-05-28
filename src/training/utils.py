import random

import numpy as np
import torch


def get_batch_single(
    obs,
    act_ids,
    act_values,
    rews,
    target_acts,
    target_length,
    batch_size,
    state_dim,
    act_value_dim,
    device,
):
    """
    target_length: int, truncate the trajectory to target_length if needed
    batch_size: int, batch size
    """

    num_trajectories, max_ep_len = obs.shape[0], obs.shape[1]
    if len(target_acts.shape) == 2:
        output_act_dim = 1
    else:
        output_act_dim = target_acts.shape[-1]

    batch_inds = np.random.choice(
        np.arange(num_trajectories),
        size=batch_size,
        replace=True,
    )

    s, a_id, a_v, r, timesteps, mask, target_a = [], [], [], [], [], [], []

    for i in range(batch_size):
        obs_i, act_id_i, act_v_i, rew_i = (
            obs[batch_inds[i]],
            act_ids[batch_inds[i]],
            act_values[batch_inds[i]],
            rews[batch_inds[i]],
        )
        si = random.randint(
            0, max_ep_len - target_length
        )  # randomly select a start index

        # get sequences from dataset
        s.append(obs_i[si : si + target_length].reshape(1, -1, state_dim))
        a_id.append(act_id_i[si : si + target_length].reshape(1, -1))
        a_v.append(
            act_v_i[si : si + target_length].reshape(1, -1, act_value_dim)
        )
        target_a.append(
            target_acts[batch_inds[i]][si : si + target_length].reshape(
                1, -1, output_act_dim
            )
        )

        r.append(rew_i[si : si + target_length].reshape(1, -1, 1))
        timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
        timesteps[-1][timesteps[-1] >= max_ep_len] = (
            max_ep_len - 1
        )  # padding cutoff

        # padding
        tlen = s[-1].shape[1]
        s[-1] = np.concatenate(
            [np.zeros((1, target_length - tlen, state_dim)), s[-1]], axis=1
        )
        a_id[-1] = np.concatenate(
            [np.zeros((1, target_length - tlen)), a_id[-1]], axis=1
        )
        a_v[-1] = np.concatenate(
            [np.zeros((1, target_length - tlen, act_value_dim)), a_v[-1]],
            axis=1,
        )
        target_a[-1] = np.concatenate(
            [np.zeros((1, target_length - tlen, output_act_dim)), target_a[-1]],
            axis=1,
        )
        r[-1] = np.concatenate(
            [np.zeros((1, target_length - tlen, 1)), r[-1]], axis=1
        )
        timesteps[-1] = np.concatenate(
            [np.zeros((1, target_length - tlen)), timesteps[-1]], axis=1
        )
        mask.append(
            np.concatenate(
                [np.zeros((1, target_length - tlen)), np.ones((1, tlen))],
                axis=1,
            )
        )

    s = torch.from_numpy(np.concatenate(s, axis=0)).to(
        dtype=torch.float32, device=device
    )
    a_id = torch.from_numpy(np.concatenate(a_id, axis=0)).to(
        dtype=torch.long, device=device
    )
    a_v = torch.from_numpy(np.concatenate(a_v, axis=0)).to(
        dtype=torch.float32, device=device
    )
    r = torch.from_numpy(np.concatenate(r, axis=0)).to(
        dtype=torch.float32, device=device
    )
    target_a = torch.from_numpy(np.concatenate(target_a, axis=0)).to(
        dtype=torch.float32, device=device
    )
    timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
        dtype=torch.long, device=device
    )
    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

    return s, a_id, a_v, r, timesteps, mask, target_a

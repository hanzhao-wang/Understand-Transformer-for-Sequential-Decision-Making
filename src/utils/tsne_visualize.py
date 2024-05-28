import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE


def tsne_embed(states, actions, rewards):
    if isinstance(states, torch.Tensor):
        states = states.detach().cpu().numpy()  # type: ignore
    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().numpy()  # type: ignore
    if isinstance(rewards, torch.Tensor):
        rewards = rewards.detach().cpu().numpy()  # type: ignore

    # concatenate states, actions, and rewards
    data = np.concatenate([states, actions, rewards], axis=-1)

    if data.ndim == 2:  # one single sample
        data = data[None, :]

    batch_size = data.shape[0]
    data = data.reshape(batch_size, -1)

    # perform t-SNE embedding
    tsne = TSNE(
        n_components=2, learning_rate="auto", init="random", random_state=42
    )
    embed = tsne.fit_transform(data)

    return embed


def plot_model_tsne(model2data):
    """
    Example:

        input = {
            "model1": {
                "states": np.array([...]),
                "actions": np.array([...]),
                "rewards": np.array([...]),
            },
            ...
        }
    """
    # ! here we assume that all the shapes are well aligned (same).

    all_states, all_actions, all_rewards, y = [], [], [], []
    idx2modelname = {}

    for midx, (model_name, data_dict) in enumerate(model2data.items()):
        states = data_dict["states"]
        actions = data_dict["actions"]
        rewards = data_dict["rewards"]

        if isinstance(states, torch.Tensor):
            states = states.detach().cpu().numpy()  # type: ignore
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()  # type: ignore
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.detach().cpu().numpy()  # type: ignore

        if states.ndim == 2:  # one single sample
            states = states[None, :]
        if actions.ndim == 2:  # one single sample
            actions = actions[None, :]
        if rewards.ndim == 2:  # one single sample
            rewards = rewards[None, :]

        batch_size = states.shape[0]
        states = states.reshape(batch_size, -1)
        actions = actions.reshape(batch_size, -1)
        rewards = rewards.reshape(batch_size, -1)

        all_states.append(states)
        all_actions.append(actions)
        all_rewards.append(rewards)
        y.append(np.ones(batch_size) * midx)
        idx2modelname[midx] = model_name

    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_rewards = np.concatenate(all_rewards, axis=0)
    y = np.concatenate(y, axis=0)

    data = np.concatenate([all_actions, all_rewards], axis=-1)

    tsne = TSNE(
        n_components=2, learning_rate="auto", init="random", random_state=42
    )

    embed = tsne.fit_transform(data)

    # do plot
    palette = np.array(sns.color_palette("hls", len(idx2modelname)))
    markers = []
    names = []
    for i, name in idx2modelname.items():
        if any([int(_) > 50 for _ in re.findall(r"\d+", name)]):
            markers.append("*")
            names.append("Step: " + str(name) + ", SelfGen Data")
        elif any([int(_) < 51 for _ in re.findall(r"\d+", name)]):
            markers.append("^")
            names.append("Step: " + str(name) + ", Raw Data")
        else:
            markers.append("o")
            names.append("Oracle Posterior")
    markers_in_model = np.array(markers)
    markers = markers_in_model[y.astype(int)]

    plt.figure(figsize=(10, 10))

    plt.scatter(
        embed[markers == "o"][:, 0],
        embed[markers == "o"][:, 1],
        lw=0,
        s=120,
        c=palette[y.astype(int)][markers == "o"],
        marker="o",
    )

    plt.scatter(
        embed[markers == "*"][:, 0],
        embed[markers == "*"][:, 1],
        lw=0,
        s=120,
        c=palette[y.astype(int)][markers == "*"],
        marker="*",
    )

    plt.scatter(
        embed[markers == "^"][:, 0],
        embed[markers == "^"][:, 1],
        lw=0,
        s=120,
        c=palette[y.astype(int)][markers == "^"],
        marker="^",
    )

    # may make some adjustment here
    # plt.xlim(-25, 25)
    # plt.ylim(-25, 25)
    # ax.axis("off")
    # ax.axis("tight")
    # add xlabel and ylabel
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    # add legend

    handles = [
        plt.scatter(
            [], [], color=palette[i], label=names[i], marker=markers_in_model[i]
        )
        for i, name in idx2modelname.items()
    ]

    # set the lengend with larger fontsize
    plt.legend(fontsize=18, markerscale=2)
    plt.show()
    plt.grid()
    plt.savefig("../figs/tsne.png")

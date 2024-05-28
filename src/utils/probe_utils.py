import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from evaluation.evaluate_episodes import evaluate_episode


class probe_NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_num):
        super(probe_NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.linear = nn.Linear(input_size, output_size)
        self.layer_num = layer_num

    def forward(self, x):
        if self.layer_num == 1:
            out = self.linear(x)
        else:
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
        return out


def train_probe(probe_model, X, y, config):
    # train the probe
    if config["loss_fn"] == "CE_loss":
        criterion = nn.CrossEntropyLoss()
    elif config["loss_fn"] == "MSE_loss":
        criterion = nn.MSELoss()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    optimizer = torch.optim.Adam(
        probe_model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=config["batch_size"],
        shuffle=True,
    )

    for epoch in range(config["max_iters"]):

        for i, (x, y) in enumerate(data_loader):
            probe_model.train()
            optimizer.zero_grad()
            outputs = probe_model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:

            # print (' Step [{}/{}],Train Loss: {:.4f}'
            #       .format( epoch,config['max_iters'], loss.item()))
            probe_model.eval()
            outputs = probe_model(X_test)
            if config["loss_fn"] == "CE_loss":
                _, predicted = torch.max(outputs.data, 1)
                acc = accuracy_score(predicted, y_test)
                val_loss = criterion(outputs, y_test)
                # print('Test Accuracy of the model on the test samples: {} %'.format(100 * acc))
                # print('Test Loss of the model on the test samples: {} '.format(val_loss.item()))
            elif config["loss_fn"] == "MSE_loss":
                acc = 0
                val_loss = criterion(outputs, y_test)
                # print('Test Loss of the model on the test samples: {} '.format(val_loss.item()))
    return acc, val_loss.item(), probe_model


def gen_samps_DP(variant):
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
    hidden_states = np.zeros(
        (
            variant["n_layer"] + 1,
            variant["num_eval_episodes"],
            variant["hidden_size"],
        )
    )
    beta_star = []
    gamma_star = []
    demand_func = []
    opt_actions = []
    demands = []
    task = variant["task"]
    with torch.no_grad():
        for i in range(variant["num_eval_episodes"]):
            _, _, hidden_state, actions_list, cum_reg, opt_action, env_info = (
                evaluate_episode(
                    task,
                    model,
                    test_env_cov=None,
                    test_mode=False,
                    need_env_info=True,  # only for probing
                    output_attention=False,
                    output_hidden_states=True,
                )
            )

            for layer in range(
                variant["n_layer"] + 1
            ):  # extra one layer as the input of GPT2
                hidden_states[layer, i, :] = (
                    hidden_state[layer][0, -1, :].detach().cpu().numpy()
                )  # choose the last token's hidden state
            beta_star.append(env_info["beta_star"])
            gamma_star.append(env_info["gamma_star"])
            demand_func.append([env_info["demand_func"]])
            opt_actions.append([env_info["opt_idx"]])
            demands.append(env_info["cur_obs"])
    beta_star = np.array(beta_star)
    gamma_star = np.array(gamma_star)
    demand_func = np.array(demand_func)
    opt_actions = np.array(opt_actions)
    demands = np.array(demands)

    # save the generated samples
    path = (
        "logs/"
        + task["name"]
        + "/"
        + task["exp_name"]
        + "/"
        + task["gen_method"]
        + "/probe/"
    )
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path + "hidden_states.npy", hidden_states)
    np.save(path + "beta_star.npy", beta_star)
    np.save(path + "gamma_star.npy", gamma_star)
    np.save(path + "demand_func.npy", demand_func)
    np.save(path + "opt_actions.npy", opt_actions)
    np.save(path + "demands.npy", demands)


def test_probe_DP(variant, config):
    task = variant["task"]
    data_path = (
        "logs/"
        + task["name"]
        + "/"
        + task["exp_name"]
        + "/"
        + task["gen_method"]
        + "/probe/"
    )
    target_list = ["demand_func", "demands", "opt_actions"]
    hidden_states = np.load(data_path + "hidden_states.npy")
    beta_star = np.load(data_path + "beta_star.npy")
    gamma_star = np.load(data_path + "gamma_star.npy")
    demand_func = np.load(data_path + "demand_func.npy")
    opt_actions = np.load(data_path + "opt_actions.npy")
    demands = np.load(data_path + "demands.npy")
    X = torch.from_numpy(hidden_states).float()
    result = {}
    for target in target_list:
        acc_list = []
        val_loss_list = []
        if target == "beta_star":
            y = torch.from_numpy(beta_star).float()
            config["loss_fn"] = "MSE_loss"
            config["output_size"] = task["state_dim"]

        elif target == "gamma_star":
            y = torch.from_numpy(gamma_star).float()
            config["loss_fn"] = "MSE_loss"
            config["output_size"] = task["state_dim"]
        elif target == "demands":
            y = torch.from_numpy(demands).float().reshape(-1, 1)
            config["loss_fn"] = "MSE_loss"
            config["output_size"] = 1
        elif target == "coefficients":
            coeffcients = np.concatenate((beta_star, gamma_star), axis=1)
            y = torch.from_numpy(coeffcients).float()
            config["loss_fn"] = "MSE_loss"
            config["output_size"] = task["state_dim"] * 2
        elif target == "demand_func":
            y = torch.from_numpy(demand_func).long().squeeze()

            config["output_size"] = 3
            config["loss_fn"] = "CE_loss"
        elif target == "opt_actions":
            y = torch.from_numpy(opt_actions).long().squeeze()
            config["loss_fn"] = "CE_loss"
            config["output_size"] = task["act_output_dim"]
        for layer in range(config["n_layer"] + 1):
            X = torch.from_numpy(hidden_states[layer, :, :]).float()

            model = probe_NN(
                config["input_size"],
                config["hidden_size"],
                config["output_size"],
                config["layer_num"],
            )
            acc, val_loss, probe_model = train_probe(model, X, y, config)
            acc_list.append(acc)
            val_loss_list.append(val_loss)
        if config["loss_fn"] == "CE_loss":
            plt.plot(1 - np.array(acc_list))
        else:
            plt.plot(val_loss_list)
        # plt.title(target)
        plt.xlabel("Layer")
        plt.ylabel("Error")
        # plt.savefig('probe/figs/'+target+'.png')
        plt.show()

        result[target] = {"acc_list": acc_list, "val_loss_list": val_loss_list}
    return result


def vis_hidden_emb_DP(variant, layer):
    task = variant["task"]
    # visualize the hidden states of layer 'layer'
    data_path = (
        "logs/"
        + task["name"]
        + "/"
        + task["exp_name"]
        + "/"
        + task["gen_method"]
        + "/probe/"
    )
    hidden_states = np.load(data_path + "hidden_states.npy")
    demand_func = np.load(data_path + "demand_func.npy")
    # use TSNE to visualize the hidden states

    X = hidden_states[layer, :500, :]
    # normalize the hidden states
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X_embedded = TSNE(n_components=2).fit_transform(X)
    # plot the hidden states with different colors according to the demand function
    plt.figure(figsize=(8, 8))
    index = np.where(demand_func[:500, 0] == 0)
    plt.scatter(
        X_embedded[index, 0], X_embedded[index, 1], c="r", label="Type1"
    )
    index = np.where(demand_func[:500, 0] == 1)
    plt.scatter(
        X_embedded[index, 0], X_embedded[index, 1], c="b", label="Type2"
    )
    index = np.where(demand_func[:500, 0] == 2)
    plt.scatter(
        X_embedded[index, 0], X_embedded[index, 1], c="g", label="Type3"
    )
    plt.legend()
    plt.xlabel("TSNE1")
    plt.ylabel("TSNE2")
    plt.title("Embedding Vectors of Layer " + str(layer))
    # plt.savefig('probe/figs/hidden_emb_layer_'+str(layer)+'.png')
    plt.show()

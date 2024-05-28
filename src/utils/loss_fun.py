import torch


def L2_loss(obs_hat, act_hat, rew_hat, obs, act, rew):
    loss = torch.nn.MSELoss()
    return loss(act_hat, act) + 0.1 * loss(rew_hat, rew)


def L2_pure_loss(obs_hat, act_hat, rew_hat, obs, act, rew):
    loss = torch.nn.MSELoss()
    return loss(act_hat, act)


def CE_loss(obs_hat, act_hat, rew_hat, obs, act, rew):
    act = act.reshape(-1).long()
    loss = torch.nn.CrossEntropyLoss()
    loss_2 = torch.nn.MSELoss()
    return loss(act_hat, act) + 0.05 * loss_2(rew_hat, rew)


def CE_pure_loss(obs_hat, act_hat, rew_hat, obs, act, rew):
    act = act.reshape(-1).long()
    loss = torch.nn.CrossEntropyLoss()
    loss_2 = torch.nn.MSELoss()
    return loss(act_hat, act)


# L1 loss
def L1_loss_pure(obs_hat, act_hat, rew_hat, obs, act, rew):
    loss = torch.nn.L1Loss()
    return loss(act_hat, act)


def L1_loss(obs_hat, act_hat, rew_hat, obs, act, rew):
    loss = torch.nn.L1Loss()
    loss2 = torch.nn.MSELoss()
    return loss(act_hat, act) + 0.1 * loss2(rew_hat, rew)


# Huber loss
def Huber_loss_pure(obs_hat, act_hat, rew_hat, obs, act, rew):
    loss = torch.nn.SmoothL1Loss()
    return loss(act_hat, act)


def Huber_loss(obs_hat, act_hat, rew_hat, obs, act, rew):
    loss = torch.nn.SmoothL1Loss()
    loss2 = torch.nn.MSELoss()
    return loss(act_hat, act) + 0.1 * loss2(rew_hat, rew)


def CE_loss_MLP(obs_hat, act_hat, rew_hat, obs, act, rew):
    act = act.reshape(-1).long()
    loss = torch.nn.CrossEntropyLoss()
    return loss(act_hat, act)

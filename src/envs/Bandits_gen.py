import numpy as np
import random
from envs.Bandits import MAB_env, LinB_env
import math
import os
import contextlib
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


def MAB_single_sample(MAB_env,method='Perturb',need_opt=True):
    MAB_env.reset()
    horizon=MAB_env.horizon
    arms_num=MAB_env.arm_space.shape[0]
    opt_rewards=[]

    act_values=[]
    act_ids=[]
    rewards=[]
    regs=[]
    opt_a=[]
    opt_flag=False
    rand_t=random.randint(0,(horizon)-1)
    for t in range(horizon):
        if t>rand_t and need_opt==True:
            if method=='Pure_exp':
                    opt_flag=True
        if method=='Perturb':
                opt_flag='Perturb'
        sel_arm=random.randint(0,arms_num-1)
      
        action_ind,reward,done,opt_arm_ind,opt_reward=MAB_env.step(sel_arm,opt_flag)

        action_value=MAB_env.arm_space[action_ind,:]
        
        act_values.append(action_value)
        act_ids.append(action_ind)
        rewards.append(reward)
        opt_a.append([opt_arm_ind])
        regs.append((opt_reward-reward))
        opt_rewards.append(opt_reward)
        if done:
                break
        
    return act_ids,act_values,rewards,regs/np.mean(opt_rewards),opt_a

def LinB_single_sample(LinB_env,method='Perturb',need_opt=True):
    LinB_env.reset()
    horizon=LinB_env.horizon
    opt_rewards=[]
    act_values=[]
    act_ids=[]
    rewards=[]
    regs=[]
    opt_a=[]
    opt_flag=False
    rand_t=random.randint(0,(horizon)-1)
    

    if method.lower() == "nips23":
        cov = np.random.choice(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        alpha = np.ones(arms_num)
        probs = np.random.dirichlet(alpha)
        probs2 = np.zeros(arms_num)
        rand_index = np.random.choice(np.arange(arms_num))
        probs2[rand_index] = 1.0
        probs = (1 - cov) * probs + cov * probs2
        for t in range(horizon):
            sel_arm = np.random.choice(np.arange(arms_num), p=probs)
            action_ind, reward, done, opt_arm_ind, opt_reward = MAB_env.step(
                sel_arm, opt_flag
            )
            action_value = MAB_env.arm_space[action_ind, :]

            act_values.append(action_value)
            act_ids.append(action_ind)
            rewards.append(reward)
            opt_a.append([opt_arm_ind])
            regs.append((opt_reward - reward))
            opt_rewards.append(opt_reward)
            if done:
                break

    else:
        for t in range(horizon):
            if t>rand_t and need_opt==True:
                if method=='Pure_exp':
                        opt_flag=True
            if method=='Perturb':
                    opt_flag='Perturb'
            sel_arm=sample_from_d_sphere(LinB_env.dim)
            action_ind,reward,done,opt_arm_ind,opt_reward=LinB_env.step(sel_arm,opt_flag)

            action_value=action_ind
            act_values.append(action_value)
            act_ids.append(0)
            rewards.append(reward)
            opt_a.append(opt_arm_ind)
            regs.append((opt_reward-reward))
            opt_rewards.append(opt_reward)
            if done:
                    break
    return act_ids,act_values,rewards,regs/np.mean(opt_rewards),opt_a

def MAB_sample_multi(exp_name=None, gen_method='Perturb', dim_num=5, num_samp=100, horizon=50, finite_envs_num=0, err_std=0.2,finenvs_seed=123):
    random.seed(13)
    np.random.seed(13)

    act_ids_list,act_values_list,rewards_list,regs_list,best_actions,states_list=[],[],[],[],[],[]
   
    

   
    if finite_envs_num!=0:
        with temp_seed(finenvs_seed):
            finite_envs_list={'theta': [np.random.normal(0,1,dim_num) for i in range(finite_envs_num)]}

    for i in range(num_samp):
        if finite_envs_num!=0:
                    theta_space=finite_envs_list['theta']
                    length=len(theta_space)
                    idx=np.random.choice(length)
                    theta_star=theta_space[idx]
        else:
                theta_star = np.random.normal(0, 1, dim_num)

        MAB_env_ = MAB_env(theta_star,horizon,err_std)
        act_ids,act_values,rewards,regs,opt_a=MAB_single_sample(MAB_env_,gen_method)

        
        regs_list.append(regs)
        act_values_list.append(act_values)
        act_ids_list.append(act_ids)
        rewards_list.append(rewards)
        best_actions.append(opt_a)

 
    act_values_list = np.array(act_values_list)
    act_ids_list = np.array(act_ids_list)
    rewards_list = np.array(rewards_list)
    regs_list = np.array(regs_list)
    best_actions = np.array(best_actions)

    states_list = np.zeros((num_samp, horizon, 1))

    data_path = "../../data/MAB/" + exp_name + "/" + gen_method
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    np.save(data_path + "/rewards.npy", rewards_list)
    np.save(data_path + "/act_values.npy", act_values_list)
    np.save(data_path + "/act_ids.npy", act_ids_list)
    np.save(data_path + "/states.npy", states_list)
    np.save(data_path + "/regs.npy", regs_list)
    np.save(data_path + "/opt_action.npy", best_actions)

    return regs_list,act_values_list, best_actions

def LinB_sample_multi(exp_name=None, gen_method='Perturb', dim_num=5, num_samp=100, horizon=50, finite_envs_num=0, err_std=0.2,finenvs_seed=123):
    random.seed(13)
    np.random.seed(13)

    act_ids_list,act_values_list,rewards_list,regs_list,best_actions,states_list=[],[],[],[],[],[]
   
    

   
    if finite_envs_num!=0:
        with temp_seed(finenvs_seed):
            finite_envs_list={'theta': [sample_from_d_sphere(dim_num) for i in range(finite_envs_num)]}

    for i in range(num_samp):
        if finite_envs_num!=0:
                    theta_space=finite_envs_list['theta']
                    length=len(theta_space)
                    idx=np.random.choice(length)
                    theta_star=theta_space[idx]
        else:
                theta_star = sample_from_d_sphere(dim_num)

        LinB_env_ = LinB_env(theta_star,horizon,err_std)
        act_ids,act_values,rewards,regs,opt_a=LinB_single_sample(LinB_env_,gen_method)

        
        regs_list.append(regs)
        act_values_list.append(act_values)
        act_ids_list.append(act_ids)
        rewards_list.append(rewards)
        best_actions.append(opt_a)

 
    act_values_list = np.array(act_values_list)
    act_ids_list = np.array(act_ids_list)
    rewards_list = np.array(rewards_list)
    regs_list = np.array(regs_list)
    best_actions = np.array(best_actions)

    states_list = np.zeros((num_samp, horizon, 1))

    data_path = "../../data/LinB/" + exp_name + "/" + gen_method
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    np.save(data_path + "/rewards.npy", rewards_list)
    np.save(data_path + "/act_values.npy", act_values_list)
    np.save(data_path + "/act_ids.npy", act_ids_list)
    np.save(data_path + "/states.npy", states_list)
    np.save(data_path + "/regs.npy", regs_list)
    np.save(data_path + "/opt_action.npy", best_actions)

    return regs_list,act_values_list, best_actions

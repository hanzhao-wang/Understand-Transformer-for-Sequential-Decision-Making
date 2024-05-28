import numpy as np
import random
import math
from numpy.linalg import inv
from numpy.linalg import norm
import os
import scipy.stats as stats
from envs.DP_ctx import DPctx_env_cts
import multiprocessing
import contextlib
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def DP_cts_sample_single(env,covariates,method='TS',need_opt=True):
    env.reset()
    horizon=env.horizon
    action_ub=env.action_ub
    opt_rewards=[]
    demands=[]
    act_values=[]
    act_ids=[]
    rewards=[]
    regs=[]
    opt_a=[]
    if method=='LinTS':
        d=int(len(covariates[0]))
        theta_hat=np.array([0.1]*(d)+[-0.1]*d)
        q_t=0
        sigma_t_inv=50*np.eye(d*2)
        sigma_t=0.02*np.eye(d*2)
        rand_t=random.randint(0,horizon-1)
        opt_flag=False
        for t in range(horizon):
                if t>rand_t:
                    opt_flag=False

                a=np.matmul(theta_hat[:d],covariates[t])
                b=np.matmul(theta_hat[d:],covariates[t])
                X_tild=np.zeros((2,2*d))
                X_tild[0,:d]=covariates[t]
                X_tild[1,d:]=covariates[t]
                M_X=X_tild@sigma_t_inv@X_tild.T
                sampling=np.random.normal(size=2)
                para=np.array([a,b])+0.8*(np.sqrt(d)/2)*np.matmul(sampling,sqrtm(M_X))
                sel_arm=np.clip(-para[0]/(2*para[1]),0,action_ub)
                demand,action_ind,reward,done,opt_arm_ind,opt_reward=env.step(sel_arm,covariates[t],opt_flag)
                action_value=action_ind
                demands.append(demand)
                act_values.append(action_value)
                act_ids.append(0)
                rewards.append(reward)
                opt_a.append(opt_arm_ind)
                regs.append((opt_reward-reward))
                opt_rewards.append(opt_reward)
                z_t=np.concatenate((covariates[t], covariates[t]*action_value))
                sigma_t=sigma_t+np.outer(z_t,z_t)
                mat_temp=np.matmul(sigma_t_inv,z_t)
                sigma_t_inv=sigma_t_inv-np.outer(mat_temp,mat_temp.T)/(1+np.matmul(z_t,mat_temp))
                q_t=q_t+z_t*demand
                theta_hat=np.matmul(sigma_t_inv,q_t)
    else:
        opt_flag=False

        rand_t=random.randint(0,(horizon)-1)
        for t in range(horizon):
            if t>rand_t and need_opt==True:
                if method=='Pure_exp':
                    opt_flag=True
            if method=='Perturb':
                    opt_flag='Perturb'
            sel_arm=np.random.uniform(0,action_ub)
            demand,action_ind,reward,done,opt_arm_ind,opt_reward=env.step(sel_arm,covariates[t],opt_flag)
            action_value=action_ind

            demands.append(demand)
            act_values.append(action_value)
            act_ids.append(0)
            rewards.append(reward)
            opt_a.append(opt_arm_ind)
            regs.append((opt_reward-reward))
            opt_rewards.append(opt_reward)
            if done:
                break
        
    return demands,act_ids,act_values,rewards,regs/np.mean(opt_rewards),opt_a

def DP_cts_sample_multi(exp_name=None, gen_method='Pure_exp', action_ub=10, dim_num=5, num_samp=100, horizon=50, finite_envs_num=0, err_std=0.2,seed=123,need_square=False):
    random.seed(13)
    np.random.seed(13)

    demands_list,act_ids_list,act_values_list,rewards_list,regs_list,best_actions=[],[],[],[],[],[]
    states_list=[]
    
    covariates=np.random.uniform(low=0,high=5,size=(num_samp,horizon,dim_num))/np.sqrt(dim_num)
    covariates[:,:,0]=1

   
    if finite_envs_num!=0:
        with temp_seed(seed):
            finite_envs_list={'beta': [(np.random.rand(dim_num)*2+1)/np.sqrt(dim_num) for i in range(finite_envs_num)],
                                'gamma':[-(np.random.rand(dim_num)+0.1)/np.sqrt(dim_num) for i in range(finite_envs_num)]}
            if need_square:
                #sample True/False for each environment
                finite_envs_list.update({'square':[True if i<(finite_envs_num/2) else False for i in range(finite_envs_num)]})
    for i in range(num_samp):
        if finite_envs_num!=0:

                    beta_space=finite_envs_list['beta']
                    length=len(beta_space)
                    idx=np.random.choice(length)
                    beta_star=beta_space[idx]
                    gamma_space=finite_envs_list['gamma']
                    gamma_star=gamma_space[idx]
                    if need_square:
                        square_space=finite_envs_list['square']
                        square=square_space[idx]
                    else:
                        square=False
        else:
                beta_star = (np.random.rand(dim_num) * 2 + 1) / np.sqrt(dim_num)
                gamma_star = -(np.random.rand(dim_num) + 0.1) / np.sqrt(dim_num)
                if need_square:
                    square=np.random.choice([True,False])
                else:
                    square=False

        demand_func=[beta_star,gamma_star]

        DP_ctx_env=DPctx_env_cts(action_ub, demand_func, horizon=horizon, err_std=err_std,square=square)

        demands,act_ids,act_values,rewards,regs,opt_a=DP_cts_sample_single(DP_ctx_env,covariates[i],method=gen_method)
        regs_list.append(regs)
        demands_list.append(demands)
        act_values_list.append(act_values)
        act_ids_list.append(act_ids)
        rewards_list.append(rewards)
        best_actions.append(opt_a)

    demands_list = np.array(demands_list)
    act_values_list = np.array(act_values_list)
    act_ids_list = np.array(act_ids_list)
    rewards_list = np.array(rewards_list)
    regs_list = np.array(regs_list)
    best_actions = np.array(best_actions)

    states_list = covariates

    data_path = "../../data/DP_ctx_cts/" + exp_name + "/" + gen_method
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    np.save(data_path + "/rewards.npy", demands_list)
    np.save(data_path + "/act_values.npy", act_values_list)
    np.save(data_path + "/act_ids.npy", act_ids_list)
    np.save(data_path + "/states.npy", states_list)
    np.save(data_path + "/regs.npy", regs_list)
    np.save(data_path + "/opt_action.npy", best_actions)

    return regs_list,act_values_list, best_actions
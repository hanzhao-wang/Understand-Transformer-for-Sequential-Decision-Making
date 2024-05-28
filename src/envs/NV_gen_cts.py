import numpy as np
import random
import math
import os
import scipy.stats as stats
from envs.NV import NV_env_cts
import contextlib
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

    

    
def NV_ctx_sample_single_cts(NV_env,covariates,method='Pure_exp'):
        NV_env.reset()
        horizon=NV_env.horizon

        states=[]
        sales=[]
        act_values=[]
        act_ids=[]
        rewards=[]
        regs=[]
        opt_a=[]

        opt_flag=False
        if method=='Pure_exp':
            rand_t=random.randint(1,horizon-1)
        opt_reward_list=[]

        for t in range(horizon):
            
            if method=='Pure_exp' and t>rand_t:
                    opt_flag=True
            if method=='Perturb':
                opt_flag='Perturb'
            cur_state=[NV_env.state,NV_env.h_b_ratio]+covariates[t].tolist() #each state is a list of current inventory, h_b_ratio,covariates
            states.append(cur_state)
            sel_order=np.random.uniform(0,NV_env.action_ub)
            tar_stock_level=sel_order+NV_env.state
            demand_larger_flag,sale,tar_order_level,opt_order_level,reward,reg,done,opt_reward=NV_env.step(tar_stock_level,covariates[t],opt_flag)
            opt_reward_list.append(opt_reward)
            action_value=tar_order_level
            action_idx=0
            #find the optimal action

            sales.append(sale)
            act_values.append(action_value)
            act_ids.append(action_idx)
            rewards.append(reward)
            opt_a.append(opt_order_level)
            regs.append(reg)
            if done:
                break
        regs=regs/np.mean(opt_reward_list)
        return states,act_ids,act_values,sales,rewards,regs,opt_a



def NV_ctx_sample_multi_cts(exp_name=None,gen_method='Pure_exp',action_ub=20,dim_num=3,num_samp=100,horizon=50,perishable=True,censor=True,finite_envs_num=0,seed=123,need_square=False):
    random.seed(13)
    np.random.seed(13)

    states_list,sales_list,act_ids_list,act_values_list,rewards_list,regs_list,best_actions=[],[],[],[],[],[],[]


    covariates=np.random.uniform(low=0,high=3,size=(num_samp,horizon,dim_num))
    covariates[:,:,0]=1
    if finite_envs_num!=0:
        with temp_seed(seed):
            finite_envs_list={'beta': [np.random.uniform(low=0,high=3,size=dim_num) for i in range(finite_envs_num)],
                                'demand_para':[np.random.uniform(low=1,high=10,size=1)[0] for i in range(finite_envs_num)],
                                }
            if need_square:
                finite_envs_list.update({'square':[True if i<(finite_envs_num/2) else False for i in range(finite_envs_num)]})
    for i in range(num_samp):

            if finite_envs_num==0:
                beta_star=np.random.uniform(low=0,high=3,size=dim_num)
                if dim_num==1:
                    beta_star=[0]
                h_b_ratio=np.random.uniform(low=0.5,high=2,size=1)[0]
                demand_para=np.random.uniform(low=1,high=10,size=1)[0]
                if need_square:
                    square=np.random.choice([True,False])
                else:
                    square=False
       
            else:
                idx=np.random.choice(finite_envs_num)
                beta_star=finite_envs_list['beta'][idx]
                if dim_num==1:
                    beta_star=[0]
                h_b_ratio=np.random.uniform(low=0.5,high=2,size=1)[0]
                demand_para=finite_envs_list['demand_para'][idx]
                if need_square:
                    square=finite_envs_list['square'][idx]
                else:
                    square=False
    
            
            demand_func={'name':'uniform','para':[0,demand_para]}
                          
            
            env=NV_env_cts(action_ub,demand_func,horizon=horizon,h_b_ratio=h_b_ratio,perishable=perishable,coeff=beta_star,censor=censor,square=square)
            states,act_ids,act_values,sales,rewards,regs,opt_a=NV_ctx_sample_single_cts(
                env,covariates[i,:,:],method=gen_method)

    
            sales_list.append(sales)
            states_list.append(states)
            act_values_list.append(act_values)
            act_ids_list.append(act_ids)
            rewards_list.append(rewards)
            best_actions.append(opt_a)
            regs_list.append(regs)


    sales_list=np.array(sales_list)
    act_values_list=np.array(act_values_list)
    act_ids_list=np.array(act_ids_list)
    rewards_list=np.array(rewards_list)
    regs_list=np.array(regs_list)
    states_list=np.array(states_list)
    best_actions=np.array(best_actions)
    data_path = "../../data/NV_cts/" + exp_name + "/" + gen_method
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    np.save(data_path +"/rewards.npy", sales_list)#save the sales as rewards
    np.save(data_path +"/act_ids.npy", act_ids_list)
    np.save(data_path +"/act_values.npy", act_values_list)
    np.save(data_path +"/states.npy", states_list)
    np.save(data_path +"/regs.npy", regs_list)
    np.save(data_path +"/opt_action.npy", best_actions)

    return regs_list,act_values_list, best_actions


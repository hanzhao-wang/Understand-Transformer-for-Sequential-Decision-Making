import numpy as np
import random
import math




class DPctx_env:
    def __init__(self,arm_space,demand_fun,horizon=100,err_std=1,env_shift=False):
        self.horizon = horizon
        self.arm_space=arm_space
        self.demand_fun=demand_fun
        self.err_std=err_std
        self.time = 0
        self.done = False
        self.opt_reward=0
        if env_shift:
            self.shift_time=np.random.randint(0,horizon/2)
            self.env_shift=True
        else:
            self.env_shift=False
            self.shift_time=1e10
        self.reset()
    def reset(self):
        self.time = 0
        self.done = False
        return self
    def step(self,action_ind,covariates,opt_flag=False):
        demand_shock=np.clip(np.random.normal(0,self.err_std),-0.5,0.5)
        demands_value=self.demand_fun(self.arm_space,covariates)
        rewards_value=self.arm_space*demands_value
    
        opt_arm_ind=np.argmax(rewards_value)
        opt_reward=rewards_value[opt_arm_ind]+demand_shock*self.arm_space[opt_arm_ind]

        if opt_flag==True:
            action_ind=opt_arm_ind
            if np.random.rand()<2/np.sqrt(self.time+1):
                action_ind=np.clip(action_ind+np.random.choice([-3,-2,-1,1,2,3]),0,len(self.arm_space)-1)
        elif opt_flag=='Perturb':
            action_ind=opt_arm_ind
            if np.random.rand()<2/np.sqrt(self.time+1):
                action_ind=np.clip(action_ind+np.random.choice([-3,-2,-1,1,2,3]),0,len(self.arm_space)-1)
        opt_reward=max(opt_reward,1e-5)
        self.opt_reward=opt_reward
        self.state=covariates
        
        demand=self.demand_fun(self.arm_space[action_ind],covariates)+demand_shock
        reward=demand*self.arm_space[action_ind]
        reward=max(reward,1e-5)
        self.time += 1
        if self.env_shift and self.time==self.shift_time:
                dim_num=int(covariates.shape[0])
                beta_star=(np.random.rand(dim_num)*2+1)/np.sqrt(dim_num)
                gamma_star=-(np.random.rand(dim_num)+0.1)/np.sqrt(dim_num)
                self.demand_fun=lambda x,covariates: beta_star@covariates+gamma_star@covariates*x
        if self.time >= self.horizon:
            self.done = True
        return  demand,action_ind,reward,self.done,opt_arm_ind,opt_reward
    

class DPctx_env_cts:
    def __init__(self,action_ub,demand_fun,horizon=100,err_std=1,square=False):
        self.horizon = horizon
        self.action_ub=action_ub
        self.alpha=demand_fun[0]
        self.beta=demand_fun[1]
        self.err_std=err_std
        self.time = 0
        self.done = False
        self.opt_reward=0
        self.reset()
        self.square=square
    def reset(self):
        self.time = 0
        self.done = False
        return self
    def step(self,action,covariates,opt_flag=False):
        demand_shock=np.clip(np.random.normal(0,self.err_std),-0.5,0.5)
        if not self.square:
            a=np.clip(self.alpha@covariates,0,1e3)
            b=np.clip(self.beta@covariates,-1e30,1e-3)
        else:
            a=np.clip(self.alpha@covariates,0,1e3)**2
            b=np.clip(-(self.beta@covariates)**2,-1e30,1e-3)
        opt_arm_ind=np.clip(-a/(2*b),0,self.action_ub)
        opt_reward=(a+b*opt_arm_ind+demand_shock)*opt_arm_ind
        action_ind=action
        if opt_flag==True:
            action_ind=opt_arm_ind
            if np.random.rand()<2/np.sqrt(self.time+1):
                action_ind=np.clip(action_ind+np.random.uniform(-1,1),0,self.action_ub)
        elif opt_flag=='Perturb':
            action_ind=opt_arm_ind
            if np.random.rand()<2/np.sqrt(self.time+1):
                action_ind=np.clip(action_ind+np.random.uniform(-0.5,0.5),0,self.action_ub)
        opt_reward=max(opt_reward,1e-5)
        self.opt_reward=opt_reward
        self.state=covariates
        
        
        demand=a+b*action_ind+demand_shock
        reward=demand*action_ind
        reward=max(reward,1e-5)
        self.time += 1
        if self.time >= self.horizon:
            self.done = True
        return  demand,action_ind,reward,self.done,opt_arm_ind,opt_reward
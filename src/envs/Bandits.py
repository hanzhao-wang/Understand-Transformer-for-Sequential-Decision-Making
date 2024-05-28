import numpy as np
import random
import math


class MAB_env:
    def __init__(self,theta,horizon=100,err_std=1):
        self.horizon = horizon
        self.theta=theta
        self.err_std=err_std
        self.time = 0
        self.done = False
        self.opt_reward=0
        self.reset()
        self.arm_space=np.eye(theta.shape[0])
    def reset(self):
        self.time = 0
        self.done = False
        return self
    def step(self,action_ind,opt_flag=False):
        noise=np.random.normal(0,self.err_std)
        opt_arm_ind=np.argmax(self.theta)
        opt_reward=self.theta[opt_arm_ind]+noise
     
        if opt_flag==True:
            action_ind=opt_arm_ind+np.random.choice([-3,-2,-1,1,2,3])
            action_ind=np.clip(action_ind,0,len(self.theta)-1)
        elif opt_flag=='Perturb':
            action_ind=opt_arm_ind
            if np.random.rand()<2/np.sqrt(self.time+1):
                action_ind=action_ind+np.random.choice([-2,-1,1,2])
                action_ind=np.clip(action_ind,0,len(self.theta)-1)

        self.opt_reward=opt_reward
        self.state=np.array(0)
        reward=self.theta[action_ind]+noise
        self.time += 1
        if self.time >= self.horizon:
            self.done = True
        return  action_ind,reward,self.done,opt_arm_ind,opt_reward


class LinB_env:
    def __init__(self,theta,horizon=100,err_std=1):
        self.horizon = horizon
        self.theta=theta
        self.err_std=err_std
        self.time = 0
        self.done = False
        self.opt_reward=0
        self.reset()
        self.dim=theta.shape[0]
    def reset(self):
        self.time = 0
        self.done = False
        return self
    def step(self,action,opt_flag=False):
        noise=np.random.normal(0,self.err_std)
        opt_arm=self.theta #in continuous case, the optimal index is the value of theta
        opt_reward=1+noise #since theta is normalized, the optimal reward is 1
      
        if opt_flag==True:
            action=opt_arm+np.random.normal(0,1)
        elif opt_flag=='Perturb':
            action=opt_arm
            if np.random.rand()<2/np.sqrt(self.time+1):
                action=action+np.random.normal(0,1)
        
        #normalize the action into unit vector
        action=action/np.linalg.norm(action)


        self.opt_reward=opt_reward
        self.state=np.array(0)
        reward=(action@self.theta).squeeze()+noise
        self.time += 1
        if self.time >= self.horizon:
            self.done = True
        return  action,reward,self.done,opt_arm,opt_reward
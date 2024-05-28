import numpy as np
import scipy.stats as stats
'''
Discrete News Vendor Environment
discrete demand function: Possion, uniform, geometric
discrete arms space/order space
'''
class NV_env:
    def __init__(self,arms_space,demand_func,horizon,h_b_ratio,perishable=True,coeff=[0],censor=True):
        self.horizon = horizon
        self.arms_space=arms_space #list of order space
        self.state=0 #inventory
        self.coeff=np.array(coeff) #coefficient for the context version if needed
        self.h_b_ratio=h_b_ratio
        self.critical_ratio=1/(1+h_b_ratio)
        self.perishable=perishable
        self.time = 0
        self.done = False
        self.demand_func=demand_func
        self.censor=censor
        self.opt_state=0 #set for checking the performance of the optimal policy

        if demand_func['name']=='possion':
            self.opt_stock_constant=np.ceil(stats.poisson.ppf(self.critical_ratio,demand_func['para']))
        elif demand_func['name']=='uniform':
            self.opt_stock_constant=np.ceil(stats.randint.ppf(self.critical_ratio, demand_func['para'][0],demand_func['para'][1]+1))# the b parameter in randint.ppf is exclusive
        elif demand_func['name']=='geometric':
            self.opt_stock_constant=np.ceil(stats.geom.ppf(self.critical_ratio,demand_func['para']))


    def reset(self):
        self.time = 0
        self.done = False
        self.state=0
        self.opt_state=0
        return self
    def step(self,tar_stock_level,ctx,opt_flag=False):
        ctx_coeff=self.coeff@ctx if self.coeff.shape[-1]>1 else 0

        if self.demand_func['name']=='possion':
            demand=stats.poisson.rvs(self.demand_func['para'])+ctx_coeff
        elif self.demand_func['name']=='uniform':
            demand=stats.randint.rvs(self.demand_func['para'][0],self.demand_func['para'][1]+1)+ctx_coeff
        elif self.demand_func['name']=='geometric':
            demand=stats.geom.rvs(self.demand_func['para'])+ctx_coeff
        
        opt_policy_order_level=np.clip(self.opt_stock_constant+ctx_coeff-self.opt_state,0,None)
        #choose the order level from arm space, which is the closest to the optimal order level but NO LESS than the optimal order
        gaps=self.arms_space-opt_policy_order_level
        #if all gaps are negative, then choose the largest one
        if np.all(gaps<0):
            opt_policy_order_idx=np.argmax(gaps)
        else:
            gaps[gaps<0]=np.inf
            opt_policy_order_idx=np.argmin(gaps)
        opt_policy_order_level=self.arms_space[opt_policy_order_idx]
        opt_stock_level=self.opt_state+opt_policy_order_level
        opt_reward=1*(demand-opt_stock_level)*(demand>opt_stock_level)+self.h_b_ratio*(opt_stock_level-demand)*(demand<=opt_stock_level)
        self.opt_state=np.clip(self.opt_state-demand+opt_policy_order_level,0,None)



        opt_order_level=np.clip(self.opt_stock_constant+ctx_coeff-self.state,0,None)
        #choose the order level from arm space
        gaps=self.arms_space-opt_order_level
        #if all gaps are negative, then choose the largest one
        if np.all(gaps<0):
            opt_order_idx=np.argmax(gaps)
        else:
            gaps[gaps<0]=np.inf
            opt_order_idx=np.argmin(gaps)
        opt_order_level=self.arms_space[opt_order_idx]

        


        tar_order_level=np.clip(tar_stock_level-self.state,0,None)
      
        if opt_flag==True:
            tar_order_level=opt_order_level
        elif opt_flag=='Perturb':
            tar_order_level=opt_order_level
            if np.random.rand()<2/np.sqrt(self.time+1):
                tar_order_level=np.clip(tar_order_level+np.random.choice([-3,-2,-1,1,2,3]),0,len(self.arms_space)-1)
        tar_stock_level=self.state+tar_order_level
        reward=1*(demand-tar_stock_level)*(demand>tar_stock_level)+self.h_b_ratio*(tar_stock_level-demand)*(demand<=tar_stock_level)
        self.state=np.clip(self.state-demand+tar_order_level,0,None)
        reg=reward-opt_reward
        demand_larger=(demand>tar_stock_level)
        sale=min(demand,tar_stock_level) if self.censor else demand

        if self.perishable:
            self.state=0
            self.opt_state=0
        self.time += 1
    
        if self.time >= self.horizon:
            self.done = True

        return  demand_larger,sale,tar_order_level,opt_order_idx,reward,reg,self.done,opt_reward
    

#continuous version
class NV_env_cts:
    def __init__(self,action_ub,demand_func,horizon,h_b_ratio,perishable=True,coeff=[0], censor=False,square=False):
        self.horizon = horizon
        self.action_ub=action_ub
        self.state=0 #inventory
        self.coeff=np.array(coeff) #coefficient for the context version if needed
        self.h_b_ratio=h_b_ratio
        self.critical_ratio=1/(1+h_b_ratio)
        self.perishable=perishable
        self.time = 0
        self.done = False
        self.demand_func=demand_func
        self.censor=censor
        self.opt_state=0 #set for checking the performance of the optimal policy
        self.square=square
        if demand_func['name']=='uniform':
            self.opt_stock_constant=stats.uniform.ppf(self.critical_ratio,self.demand_func['para'][0],self.demand_func['para'][1])

    def reset(self):
        self.time = 0
        self.done = False
        self.state=0
        self.opt_state=0
        return self
    def step(self,tar_stock_level,ctx,opt_flag=False):
        if not self.square:
            ctx_coeff=self.coeff@ctx if self.coeff.shape[-1]>1 else 0
        else:
            ctx_coeff=(self.coeff@ctx)**2 if self.coeff.shape[-1]>1 else 0

        if self.demand_func['name']=='uniform':
            demand=stats.uniform.rvs(self.demand_func['para'][0],self.demand_func['para'][1])+ctx_coeff
        opt_policy_order_level=np.clip(self.opt_stock_constant+ctx_coeff-self.opt_state,0,self.action_ub)
        opt_policy_order_idx=0
       
        opt_stock_level=self.opt_state+opt_policy_order_level
        opt_reward=1*(demand-opt_stock_level)*(demand>opt_stock_level)+self.h_b_ratio*(opt_stock_level-demand)*(demand<=opt_stock_level)
        self.opt_state=np.clip(self.opt_state-demand+opt_policy_order_level,0,None)


        opt_order_level=np.clip(self.opt_stock_constant+ctx_coeff-self.state,0,self.action_ub)
        opt_order_idx=0
        tar_order_level=np.clip(tar_stock_level-self.state,0,self.action_ub)
      
        if opt_flag==True:
            tar_order_level=opt_order_level
        elif opt_flag=='Perturb':
            tar_order_level=opt_order_level
            per_level=2/np.sqrt(self.time+1)
            tar_order_level=np.clip(tar_order_level+np.random.uniform(-per_level,per_level),0,self.action_ub)
        tar_stock_level=self.state+tar_order_level
        reward=1*(demand-tar_stock_level)*(demand>tar_stock_level)+self.h_b_ratio*(tar_stock_level-demand)*(demand<=tar_stock_level)
        self.state=np.clip(self.state-demand+tar_order_level,0,None)
        reg=reward-opt_reward
        demand_larger=(demand>tar_stock_level)
        sale=min(demand,tar_stock_level) if self.censor else demand

        if self.perishable:
            self.state=0
            self.opt_state=0
        self.time += 1
    
        if self.time >= self.horizon:
            self.done = True
        return  demand_larger,sale,tar_order_level,opt_order_level,reward,reg,self.done,opt_reward
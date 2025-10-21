from algos.common.network_base import MLP, initWeights
from algos.common.actor_base import (
    ActorBase, normalize, unnormalize, clip
)

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import os
import asyncio
import websockets
import json
import threading



class ActorGaussian(ActorBase):
    def __init__(self, device:torch.device, obs_dim:int, action_dim:int, \
                action_bound_min:np.ndarray, action_bound_max:np.ndarray, actor_cfg:dict, \
                log_std_min:float=-4.0, log_std_max:float=2.0) -> None:
        ActorBase.__init__(self, device)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # for model
        self.actor_cfg = actor_cfg
        self.use_action_bound = self.actor_cfg['use_action_bound']
        if 'last_activation' in self.actor_cfg:
            self.last_activation = eval(f'torch.nn.{self.actor_cfg["last_activation"]}')()
        else:
            self.last_activation = lambda x: x
        self.activation = eval(f'torch.nn.{self.actor_cfg["mlp"]["activation"]}')
        self.log_std_init = self.actor_cfg['log_std_init']
        self.log_std_fix = self.actor_cfg['log_std_fix']

        # for action
        if self.use_action_bound:
            self.action_bound_min = torch.tensor(
                action_bound_min, device=device, dtype=torch.float32)
            self.action_bound_max = torch.tensor(
                action_bound_max, device=device, dtype=torch.float32)
        else:
            self.action_bound_min = torch.tensor(
                -np.ones(self.action_dim), device=device, dtype=torch.float32)
            self.action_bound_max = torch.tensor(
                np.ones(self.action_dim), device=device, dtype=torch.float32)

        # build model
        self.build()


    def build(self) -> None:
        self.add_module('model', MLP(
            input_size=self.obs_dim, output_size=self.actor_cfg['mlp']['shape'][-1], \
            shape=self.actor_cfg['mlp']['shape'][:-1], activation=self.activation,
        ))
        self.add_module("mean_decoder", torch.nn.Sequential(
            self.activation(),
            torch.nn.Linear(self.actor_cfg['mlp']['shape'][-1], self.action_dim),
        ))
        if self.log_std_fix:
            self.std_decoder = lambda x: torch.ones(
                *x.shape[:-1], self.action_dim, dtype=torch.float, device=self.device)*self.log_std_init
        else:
            self.add_module("std_decoder", torch.nn.Sequential(
                self.activation(),
                torch.nn.Linear(self.actor_cfg['mlp']['shape'][-1], self.action_dim),
            ))
        
    def forward(self, state:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        output: (mean, log_std, std)
        '''
        x = self.model(state)
        mean = self.last_activation(self.mean_decoder(x))
        log_std = self.std_decoder(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, log_std, std

    def updateActionDist(self, state:torch.Tensor, epsilon:torch.Tensor) -> None:
        self.action_mean, self.action_log_std, self.action_std = \
            self.forward(state)
        self.normal_action = self.action_mean + epsilon*self.action_std
        self.action_dist = torch.distributions.Normal(self.action_mean, self.action_std)

    def sample(self, deterministic:bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        if deterministic:
            norm_action = self.action_mean
        else:
            norm_action = self.normal_action
        unnorm_action = unnormalize(norm_action, self.action_bound_min, self.action_bound_max)
        return norm_action, unnorm_action

    def getMeanStd(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.action_mean, self.action_std
    
    def getDist(self) -> torch.distributions.Distribution:
        return self.action_dist
        
    def getEntropy(self) -> torch.Tensor:
        '''
        return entropy of action distribution given state.
        '''
        entropy = torch.mean(torch.sum(self.action_dist.entropy(), dim=-1)) + 0.0*self.action_mean.mean()
        return entropy
    
    def getLogProb(self, action=None) -> torch.Tensor:
        '''
        return log probability of action given state.
        '''
        if action is None:
            log_prob = torch.sum(self.action_dist.log_prob(self.normal_action), dim=-1)
        else:
            log_prob = torch.sum(self.action_dist.log_prob(action), dim=-1)
        return log_prob

    def initialize(self) -> None:
        for name, module in self.named_children():
            if name == 'std_decoder':
                initializer = lambda m: initWeights(m, init_bias=self.log_std_init)
            else:
                initializer = lambda m: initWeights(m)
            module.apply(initializer)

# class ActorGaussian(ActorBase):
#     def __init__(self, device:torch.device, obs_dim:int, action_dim:int, \
#                 action_bound_min:np.ndarray, action_bound_max:np.ndarray, actor_cfg:dict, \
#                 log_std_min:float=-4.0, log_std_max:float=2.0) -> None:
#         ActorBase.__init__(self, device)

#         self.obs_dim = obs_dim
#         self.action_dim = action_dim
#         self.log_std_min = log_std_min
#         self.log_std_max = log_std_max
#         self.i = 0
#         # for model
#         self.actor_cfg = actor_cfg
#         self.use_action_bound = self.actor_cfg['use_action_bound']
#         if 'last_activation' in self.actor_cfg:
#             self.last_activation = eval(f'torch.nn.{self.actor_cfg["last_activation"]}')()
#         else:
#             self.last_activation = lambda x: x
#         self.activation = eval(f'torch.nn.{self.actor_cfg["mlp"]["activation"]}')
#         self.log_std_init = self.actor_cfg['log_std_init']
#         self.log_std_fix = self.actor_cfg['log_std_fix']

#         # for action
#         if self.use_action_bound:
#             self.action_bound_min = torch.tensor(
#                 action_bound_min, device=device, dtype=torch.float32)
#             self.action_bound_max = torch.tensor(
#                 action_bound_max, device=device, dtype=torch.float32)
#         else:
#             self.action_bound_min = torch.tensor(
#                 -np.ones(self.action_dim), device=device, dtype=torch.float32)
#             self.action_bound_max = torch.tensor(
#                 np.ones(self.action_dim), device=device, dtype=torch.float32)
#         # self.ws = None
#         # self.connect()
#         # self.loop = asyncio.new_event_loop()
#         # threading.Thread(target=self.loop.run_forever, daemon=True).start()
#         # asyncio.run_coroutine_threadsafe(self.connect(), self.loop)
#         # build model
#         self.build()

#     async def connect(self):
#         self.ws = await websockets.connect("ws://10.112.148.127:8765")
#         print("✅ 已连接到服务器")

#     def forward(self, state:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         '''
#         output: (mean, log_std, std)
#         '''
#         x = self.model(state)
#         mean = self.last_activation(self.mean_decoder(x))

#         # noise = np.random.choice([-10, 10], size=mean[0].shape)
#         # mean = mean + torch.from_numpy(noise).to('cuda:0')
#         # val = -0.5
#         # mean = torch.tensor([val, val, val, val, val, val, val, val, val, val, val, val]).to('cuda:0')

#         # state_np = state.cpu().numpy()
#         # state_list = state_np.tolist()
#         # future = asyncio.run_coroutine_threadsafe(self.ws.send(json.dumps(state_list)), self.loop)
#         # resp = asyncio.run_coroutine_threadsafe(self.ws.recv(), self.loop).result()
#         # resp_list = json.loads(resp)  # 变成 Python 嵌套列表
#         # resp_np = np.array(resp_list, dtype=np.float32)
#         # resp_tensor = torch.from_numpy(resp_np).to('cuda:0')
#         # mean = resp_tensor

#         # np_array = mean.cpu().numpy()
#         # save_path = os.path.join("/home/wunuo/Workspace/Stage-Wise-CMORL/val_data/model_output/bpu", f"val_data_{self.i:03d}.npy")
#         # np.save(save_path, np_array)
#         # self.i = self.i + 1


#         log_std = self.std_decoder(x)
#         log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
#         std = torch.exp(log_std)
#         # print(mean)
#         return mean, log_std, std

#     def build(self) -> None:
#         self.add_module('model', MLP(
#             input_size=self.obs_dim, output_size=self.actor_cfg['mlp']['shape'][-1], \
#             shape=self.actor_cfg['mlp']['shape'][:-1], activation=self.activation,
#         ))
#         self.add_module("mean_decoder", torch.nn.Sequential(
#             self.activation(),
#             torch.nn.Linear(self.actor_cfg['mlp']['shape'][-1], self.action_dim),
#         ))
#         if self.log_std_fix:
#             self.std_decoder = lambda x: torch.ones(
#                 *x.shape[:-1], self.action_dim, dtype=torch.float, device=self.device)*self.log_std_init
#         else:
#             self.add_module("std_decoder", torch.nn.Sequential(
#                 self.activation(),
#                 torch.nn.Linear(self.actor_cfg['mlp']['shape'][-1], self.action_dim),
#             ))

#     def updateActionDist(self, state:torch.Tensor, epsilon:torch.Tensor) -> None:
#         self.action_mean, self.action_log_std, self.action_std = \
#             self.forward(state)

#         # #save input data
#         # np_array = state.cpu().numpy()
#         # save_path = os.path.join("/home/wunuo/Workspace/Stage-Wise-CMORL/cal_data/twohand", f"cal_data_{self.i:03d}.npy")
#         # np.save(save_path, np_array)
#         # self.i = self.i + 1

#         self.normal_action = self.action_mean + epsilon*self.action_std
#         self.action_dist = torch.distributions.Normal(self.action_mean, self.action_std)

#     def sample(self, deterministic:bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
#         if deterministic:
#             norm_action = self.action_mean
#         else:
#             norm_action = self.normal_action
#         unnorm_action = unnormalize(norm_action, self.action_bound_min, self.action_bound_max)
#         return norm_action, unnorm_action

#     def getMeanStd(self) -> Tuple[torch.Tensor, torch.Tensor]:
#         return self.action_mean, self.action_std
    
#     def getDist(self) -> torch.distributions.Distribution:
#         return self.action_dist
        
#     def getEntropy(self) -> torch.Tensor:
#         '''
#         return entropy of action distribution given state.
#         '''
#         entropy = torch.mean(torch.sum(self.action_dist.entropy(), dim=-1)) + 0.0*self.action_mean.mean()
#         return entropy
    
#     def getLogProb(self, action=None) -> torch.Tensor:
#         '''
#         return log probability of action given state.
#         '''
#         if action is None:
#             log_prob = torch.sum(self.action_dist.log_prob(self.normal_action), dim=-1)
#         else:
#             log_prob = torch.sum(self.action_dist.log_prob(action), dim=-1)
#         return log_prob

#     def initialize(self) -> None:
#         for name, module in self.named_children():
#             if name == 'std_decoder':
#                 initializer = lambda m: initWeights(m, init_bias=self.log_std_init)
#             else:
#                 initializer = lambda m: initWeights(m)
#             module.apply(initializer)

class ActorGaussianWrapper(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.model = actor.model
        self.mean_decoder = actor.mean_decoder
        self.last_activation = actor.last_activation

    def forward(self, state):
        x = self.model(state)
        mean = self.last_activation(self.mean_decoder(x))
        return mean

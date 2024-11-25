# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,#策略价值网络
                 num_learning_epochs=1,#每次更新策略时的学习轮数
                 num_mini_batches=1,#每轮训练的mini-batch数量，用于分割采样的数据
                 clip_param=0.2,#PPO中的剪辑参数，用于限制策略更新幅度，避免更新过大
                 gamma=0.998,#discounted rate，折扣因子
                 lam=0.95,#GAE(广义优势估计)中用来平衡偏差和方差的系数
                 value_loss_coef=1.0,#价值函数损失的权重系数，用于平衡actor和critic的损失
                 entropy_coef=0.0,#策略熵的系数，用于鼓励探索
                 learning_rate=1e-3,#学习率，控制参数更新的步幅
                 max_grad_norm=1.0,#梯度裁剪的最大值，防止梯度爆炸
                 use_clipped_value_loss=True,#是否使用剪辑的价值损失，帮助稳定critic的更新
                 schedule="fixed",#学习率调整策略(例如固定学习率或自适应调整)
                 desired_kl=0.01,#目标KL散度，用于控制策略更新的幅度
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)#Adam 优化器，负责更新 actor_critic 的参数
        self.transition = RolloutStorage.Transition()#初始化采样的存储

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:#检查 actor_critic 模型是否为递归神经网络（RNN）结构，如果是，那么需要记录当前的隐状态（hidden states）
            self.transition.hidden_states = self.actor_critic.get_hidden_states()#这个函数只有在ActorCriticRecurrent类中定义
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()#返回一个以actor网络输出的aciton为均值，设定的std为标准差的正态分布的采样
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()#返回critic网络的值输出
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()#获得所有action的对数概率之和，即整个动作的联合对数概率
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)#将该步的数据添加到storage中
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:


                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])#输入actor网络，获得当前策略下的action输出，actor网络的前向传播
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])#获得当前的值，critic网络的前向传播
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():#临时禁用梯度计算
                        kl = torch.sum(#求取分布N(μ,σ^2) 和 N(old_μ,old_^2)的KL散度，即新旧策略的KL散度，最后的形状是（mini_batch_size*1）
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)#对mini_batch_size个kl散度求均值得到标量

                        #通过kl散度来动态调整学习率
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss，优化器的目的是最小化损失，而RL的目标是最大化目标函数，因此下面的损失函数跟目标函数正好是反的，不仅元素加了负号，而且min变成了max
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))#获得新旧策略概率之比
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()#目标函数值的相反数，或者叫做Surrogate loss

                # Value function loss，将模型的值预测与实际回报（returns_batch）对齐
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)#target_values_batch实际上是old_value，因此这里是将value_batch的更新限制了
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()#总损失为目标函数损失、带系数的值损失和熵损失的均值构成。其中熵的项为负数，则增大熵会减少总损失，鼓励策略更多样化

                # Gradient step
                self.optimizer.zero_grad()#清除（归零）模型中所有参数的梯度
                loss.backward()#计算损失 loss 对模型所有参数的梯度
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)#使用梯度裁剪技术来控制梯度的最大范数。这一步通过限制梯度的范数大小来防止梯度爆炸，从而确保训练稳定
                self.optimizer.step()#用优化器 self.optimizer（例如 SGD、Adam 等）执行梯度更新

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches#获得损失计算的执行步骤数量
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss

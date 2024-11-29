from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float

class Tita(LeggedRobot):

    def _compute_torques(self, actions):
        
        actions_scaled = actions * self.cfg.control.action_scale
        dof_pos_ref_left = actions[:,:3]* self.cfg.control.action_scale
        dof_vel_ref_left = actions[:,3]* self.cfg.control.action_scale*20
        dof_pos_ref_right = actions[:,4:7]* self.cfg.control.action_scale
        dof_vel_ref_right = actions[:,7]* self.cfg.control.action_scale*20

        torques_pos = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        torques_left_pos = torques_pos[:,:3]
        torques_right_pos = torques_pos[:,4:7]
        dof_vel_gain = self.d_gains[0]
        dof_vel_gain.repeat(self.num_envs)

        target_vel = self.commands[:, 0]
        target_w = self.commands[:, 1]
        left_wheel_vel_target = target_vel/0.09 - target_w
        right_wheel_vel_target = -target_vel/0.09 - target_w
        torques_left_vel = dof_vel_gain*(dof_vel_ref_left   - self.dof_vel[:,3]) 
        torques_right_vel = dof_vel_gain*(dof_vel_ref_right  - self.dof_vel[:,7]) 

        torques = torch.cat(
            (
                torques_left_pos,
                torques_left_vel.unsqueeze(1),
                torques_right_pos,
                torques_right_vel.unsqueeze(1)
            ),
            axis = 1,
                            )
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)#对需要重新采样的机器人随即采样新的目标命令
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)#通过四元数获取机体x轴在世界坐标系下的向量坐标
            heading = torch.atan2(forward[:, 1], forward[:, 0])#计算航向角
            self.commands[:, 1] = torch.clip(1.5*wrap_to_pi(self.commands[:, 3] - heading), -5., 5.)#通过目标航向角和当前航向角的误差获取角速度，类似PID

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):#每执行push_interval次仿真步后生效
            self._push_robots()#在仿真环境中模拟随机外力对机器人的影响
        self.still_flag = (torch.abs(self.commands[:,0]) < 0.1) & (torch.abs(self.commands[:,1]) < 0.1)


    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["height"][0], self.command_ranges["height"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

    def check_termination(self):
        """ Check if environments need to be reset
        """
        termination_contact_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        overturning_buf = (self.projected_gravity[:, 2] > -0.0)
        delay_termination_buf_step = (termination_contact_buf | overturning_buf)
        self.delay_termination_buf += delay_termination_buf_step
        self.reset_buf = (self.delay_termination_buf > self.cfg.env.delay_termination_time_s / self.dt)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def compute_observations(self):
        """ Computes observations
        """
        #torch.cat 将多个张量沿最后一个维度拼接在一起，乘以观测尺度是为了确保观测数据在处理和学习中的有效性和一致性，从而提高整体性能
        self.obs_buf = torch.cat((  #self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        if self.cfg.env.num_privileged_obs is not None:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat(
                (
                    self.base_lin_vel * self.obs_scales.lin_vel,
                    self.obs_buf,
                    heights,
                    self.contact_forces[:, self.feet_indices,0],
                    self.contact_forces[:, self.feet_indices,1],
                    self.contact_forces[:, self.feet_indices,2]
                ),
                dim=-1,
            )
            
        # add noise if needed
        if self.add_noise:
            #torch.rand_like(self.obs_buf)：这个函数生成一个与 self.obs_buf 形状相同的张量，其中的值都是在 [0, 1) 范围内均匀分布的随机数
            #2 * torch.rand_like(self.obs_buf) - 1：通过将生成的随机数乘以 2，然后减去 1，将随机数的范围转换为 [-1, 1)
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise#是否添加噪声的标志
        noise_scales = self.cfg.noise.noise_scales#不同类型噪声的缩放因子
        noise_level = self.cfg.noise.noise_level#噪声的整体强度

        # noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:17] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[17:25] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[25:33] = 0. # previous actions

        return noise_vec

    def _init_buffers(self):
        super()._init_buffers()
        self.delay_termination_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.theta_thigh_left = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.theta_thigh_right = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.theta_left = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.theta_right = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.theta_hip_left = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.theta_hip_right = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.still_flag = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device, requires_grad=False)
        self.still_gain = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.still_gain[:,0] = 1
        self.still_gain[:,4] = 1
        self.last_actions = torch.zeros(self.num_envs,self.num_actions,2,dtype=torch.float,device=self.device,requires_grad=False)
        self.is_step = torch.zeros(self.num_envs,1,dtype=torch.bool,device=self.device,requires_grad=False)
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.delay_termination_buf[env_ids] = 0.
    
    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        # 计算相邻点的高度差
        height_diff_1 = torch.abs(heights1 - heights2)
        height_diff_2 = torch.abs(heights1 - heights3)
        height_diff = torch.max(height_diff_1, height_diff_2)

        # 设定阈值来区分台阶和斜坡
        step_threshold = 0.2  # 根据实际情况调整阈值
        is_step = height_diff > step_threshold

        # 对每个环境，检查是否有任何点被识别为台阶
        self.is_step = torch.any(is_step.view(self.num_envs, -1), dim=1).view(-1, 1)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:,:,0] - self.actions), dim=1)

    def _reward_action_smooth(self):
        # Penalize changes in actions
        return (torch.sum(torch.square(self.actions[:, :3]- 2 * self.last_actions[:, :3, 0]+ self.last_actions[:, :3, 1]),dim=1) +  
               torch.sum(torch.square(self.actions[:, 4:7]- 2 * self.last_actions[:, 4:7, 0]+ self.last_actions[:, 4:7, 1]),dim=1))
         
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos)*self.still_gain, dim=1)*self.still_flag
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 1] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_no_fly(self):  # todo重载奖励函数
        contacts_xy = torch.norm(self.contact_forces[:, self.feet_indices,:2],dim=2)  > 0.5
        contacts_z = self.contact_forces[:, self.feet_indices,2]  > 0.5
        vertical_flag = torch.sum(1.0 * contacts_xy, dim=1) > 0
        vertical_flag = vertical_flag & self.is_step.view(-1) 
        output = torch.where(vertical_flag, torch.tensor(1.0), torch.tensor(-1.0))
        theta_slope = torch.atan2(contacts_xy,contacts_z)
        slope_flag = torch.any(theta_slope) < 3.14/4
        # rew = (-1.0*(torch.sum(1.0 * contacts_z, dim=1) == 2)*output   
        #     + 2.0*(torch.sum(1.0 * contacts_z, dim=1) > 0)
        #     + 1.0*(torch.sum(1.0 * contacts_z, dim=1) == 1)*output)
        rew = 1.0*(torch.sum(1.0 * contacts_z, dim=1) > 0)
        return rew

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        height_err = torch.square(self.commands[:,2] - base_height)
        return torch.exp(-height_err / self.cfg.rewards.tracking_sigma)
        # return height_err

    def _reward_no_moonwalk(self):
        x = self.base_quat[:, 0]
        y = self.base_quat[:, 1]
        z = self.base_quat[:, 2]
        w = self.base_quat[:, 3]
        pitch = torch.arcsin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))

        joints = list(self.cfg.init_state.default_joint_angles.keys())

        left_thigh_angle = joints.index("joint_left_thigh")
        left_calf_angle = joints.index("joint_left_calf")
        left_hip_angle = joints.index("joint_left_hip")

        right_thigh_angle = joints.index("joint_right_thigh")
        right_calf_angle = joints.index("joint_right_calf")
        right_hip_angle = joints.index("joint_right_hip")

        # 轮子同步时两连杆在x轴方向投影之和相同
        len_ratio = 1  # 连杆长度之比
        self.theta_thigh_left = self.dof_pos[:,left_thigh_angle]
        self.theta_thigh_right = -self.dof_pos[:,right_thigh_angle]

        theta_thigh2vertical_left = self.theta_thigh_left + pitch
        theta_thigh2vertical_right = self.theta_thigh_right + pitch

        self.theta_left = 3.14-(-self.dof_pos[:, left_calf_angle])
        self.theta_right = 3.14-self.dof_pos[:, right_calf_angle]
        
        self.theta_hip_left = self.dof_pos[:,left_hip_angle]
        self.theta_hip_right = self.dof_pos[:,right_hip_angle]

        l = torch.torch.sin(theta_thigh2vertical_left+self.theta_left)-len_ratio*torch.torch.sin(theta_thigh2vertical_left)
        r = -(torch.torch.sin(theta_thigh2vertical_right+self.theta_right)-len_ratio*torch.torch.sin(theta_thigh2vertical_right))
        
        rew = torch.square(r + l)
        return rew

    def _reward_hip_angle(self):
        x = self.base_quat[:, 0]
        y = self.base_quat[:, 1]
        z = self.base_quat[:, 2]
        w = self.base_quat[:, 3]
        roll = torch.atan2(2.0 * (w *z + x * y), 1.0 - 2.0 * (y**2 + z**2))

        theta_wheel_rol_left = self.theta_hip_left - roll
        theta_wheel_rol_right = self.theta_hip_right - roll

        hip2rol_err = (torch.mean(torch.square(theta_wheel_rol_left)) + torch.mean(torch.square(theta_wheel_rol_right)))/2
        return hip2rol_err

    def _calculate_heigh(self):
        x = self.base_quat[:, 0]
        y = self.base_quat[:, 1]
        z = self.base_quat[:, 2]
        w = self.base_quat[:, 3]

        pitch = torch.arcsin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
        roll = torch.atan2(2.0 * (w *x + y * z), 1.0 - 2.0 * (x**2 + y**2))

        theta_wheel_rol_left = self.theta_hip_left + roll
        theta_left_thigh2vertical = self.theta_thigh_left - pitch
        theta_left_hight = theta_left_thigh2vertical + self.theta_left/2 - 1.57
        l_left = 0.20*torch.sin(self.theta_left/2)*2
        left_hight = l_left*torch.cos(theta_left_hight) + 0.08

        theta_wheel_rol_right = -self.theta_hip_right - roll
        theta_right_thigh2vertical = self.theta_thigh_right - pitch 
        theta_right_hight = theta_right_thigh2vertical + self.theta_right/2 - 1.57
        l_right = 0.20*torch.sin(self.theta_right/2)*2
        right_hight = l_right*torch.cos(theta_right_hight) + 0.08

        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        left_hegiht_err = torch.mean(base_height - left_hight*torch.cos(theta_wheel_rol_left))
        right_height_err = torch.mean(base_height - right_hight*torch.cos(theta_wheel_rol_right))

        rew = torch.mean(torch.square(theta_wheel_rol_left)) + torch.mean(torch.square(theta_wheel_rol_right))
        contacts_xy = torch.norm(self.contact_forces[:, self.feet_indices,:2],dim=2)

        height_err = torch.square(self.commands[:,2] - base_height)*~self.still_flag

        contacts_xy = torch.norm(self.contact_forces[:, self.feet_indices,:2],dim=2)  > 0.5
        contacts_z = self.contact_forces[:, self.feet_indices,2]  > 0.1
        vertical_flag = torch.sum(1.0 * contacts_xy, dim=1) > 0
        vertical_flag = vertical_flag & self.is_step.view(-1) 

        true_count = vertical_flag.sum()
        true_ratio = true_count.float() / vertical_flag.numel()
        output = torch.where(vertical_flag, torch.tensor(1.0), torch.tensor(-1.0))

        print("base_height = ",base_height)
        print("target_height = ",self.commands[:,2] )
        print("right_hight = ", right_hight)
        print("left_hight = ",left_hight)
        print("theta_wheel_rol_left",self.theta_hip_left)
        print("theta_wheel_rol_right",self.theta_hip_right)
        print("height_err",height_err)
        print("true_ratio=",true_ratio )
        print("self.commands[:,0]",self.commands[:,0])
        print("self.commands[:,1]",self.commands[:,1])
        print("self.base_vel",self.base_lin_vel[:,0])
        

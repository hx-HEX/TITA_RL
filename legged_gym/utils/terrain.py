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

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]#不同类型地形所占比例

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols#num_rows表示地形的等级level，num_cols表示地形的种类type，这个变量就表示子地形的总数
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))#存储了所有子地形的起点

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)#一种地形的宽度栅格大小
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)#一种地形的长度栅格大小

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)#缓冲区边长
        #整个地形包括所有相互连接的子地形和最外面包围的缓冲区
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)#存储整个地形的高度信息，只有各个子地形会被更新，缓冲区边界默认为0

        #根据标志位选择如何生成地形（按照对应的系数、按照随机生成的系数、只生成一种）
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        #传入栅格地图的高度信息
        self.heightsamples = self.height_field_raw
        #将高度场（heightfield）转换为三角网格（TriMesh），以便在图形渲染、物理仿真等应用中更有效地表示和处理地形。返回的是
        #self.vertices: 转换后的三维顶点列表，表示地形表面的所有点（顶点）。
        #self.triangles: 转换后的三角形索引列表，每个三角形由三个顶点索引组成，表示如何连接这些顶点来形成三角形面片
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    #每个子地形的类型系数(0-1)、难度系数随机生成(三选一)，之后再按照生成的系数生成地形
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))#将一维的索引k转换为对应的二维索引(i,j)，其中一维数组大小为num_sub_terrains，二维数组大小为(num_rows,num_cols)

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])#随机选择其中一个
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
    
    #按照子地形对应的难度系数、类型系数生成地形，系数都在0-1之间
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows#rows对应level，也就是地形难度等级
                choice = j / self.cfg.num_cols + 0.001#cols对应types，也就是选择的种类

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    #按照事先选定的类型生成地形
    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            #调用相应种类地形的生成函数，前提是函数“terrain_type”被定义
            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)#左边的括号为函数名，右边的括号为传入的参数
            self.add_terrain_to_map(terrain, i, j)
    
    #根据传入的地形类型系数和难度系数生成对应的地形
    def make_terrain(self, choice, difficulty):
        #创建子地形
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        #通过传入的地形难度系数(difficulty)更新斜坡坡度、台阶高度、障碍物高度、踏石大小、空隙大小、坑洞深度
        slope = difficulty * 0.4
        step_height = 0.02 + 0.1 * difficulty
        discrete_obstacles_height = 0.02 + difficulty * 0.02
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        
        #通过传入的类型系数确定生成的地形，前6种地形直接调用terrain_utils的api生成，后2种自定义生成，本质上是生成terrain.height_field_raw
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1#下坡
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)#坡地
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)#坡地加随机高度
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1#下台阶
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)#台阶
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)#障碍物地形
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)#踏石地形
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)#空隙地形
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)#坑洞地形
        
        return terrain

    #将子地形加入到地图中
    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system，地图高度信息的子地形索引要加上过渡区的偏置self.border
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        #子地形在整个地图的中点坐标（不包含过渡区，因此没有加偏置，但在后续用到高度时索引需要加偏置）
        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width

        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale#取以地形中点为中心的2m*2m矩形范围内的最高点
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

#自定义空隙和坑洞的形状
def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

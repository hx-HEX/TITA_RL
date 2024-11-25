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

#这段代码的作用是自动初始化一个类及其所有嵌套的成员类实例。
#通过这种方式，可以确保在创建 BaseConfig 的实例时，所有嵌套的类都会被正确实例化。这在需要复杂的配置类时非常有用，能够减少手动初始化的工作

#inspect类用于检查对象的类型和属性
import inspect

class BaseConfig:
    #__init__ 方法是类的构造函数，当实例化 BaseConfig 时会被调用。
    #在初始化过程中，调用 init_member_classes 方法，传入当前实例 self
    def __init__(self) -> None:
        """ Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods)."""
        self.init_member_classes(self)
    
    #init_member_classes 是一个静态方法，它接受一个对象 obj 作为参数。
    @staticmethod
    def init_member_classes(obj):
        # iterate over all attributes names,使用 dir() 函数获取 obj 的所有属性和方法名称。
        for key in dir(obj):
            # disregard builtin attributes
            # if key.startswith("__"):
            #如果属性名称是 __class__，则跳过该属性。注释中提到可以忽略以 __ 开头的属性，这里实际上只排除了 __class__
            if key=="__class__":
                continue
            # get the corresponding attribute object,使用 getattr() 函数获取属性对象
            var =  getattr(obj, key)
            # check if it the attribute is a class,使用 inspect.isclass() 检查 var 是否是一个类
            if inspect.isclass(var):
                # instantate the class,如果是类，则实例化该类，创建一个对象 i_var
                i_var = var()
                # set the attribute to the instance instead of the type,将原来的属性设置为新实例 i_var
                setattr(obj, key, i_var)
                # recursively init members of the attribute,对新实例调用 init_member_classes，递归地初始化它的成员类
                BaseConfig.init_member_classes(i_var)
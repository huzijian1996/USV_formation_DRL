# -*- coding: utf-8 -*-

import numpy as np
import math
from director import vtkAll as vtk
from director import ioUtils, filterUtils
from sensor import RaySensor

np.random.seed(1)

#智能体(被控体)
class Robot(object):
    #创建一个Agent
    def __init__(self, robot_name="USV", init_state=[145, 145, 0], velocity=15):
        #初始化
        self._max_turnangle_perstep = np.pi / 30  # 智能体每个步长最大的偏转角度6 degree
        self._dt = 0.1
        self.state = np.array([0., 0., 0.])
        self.a_max = 40
        #固定翼飞机A10
        if robot_name == "USV":
            model = "USV1.obj"

        else:  #此处留待以后增加其他模型
            model = "USV1.obj"

        #显示转换
        t = vtk.vtkTransform()
        t.Scale(0.1, 0.1, 0.1)
        polydata = ioUtils.readPolyData(model)
        self._raw_polydata = filterUtils.transformPolyData(polydata, t)

        #初始化
        # self.velocity = velocity
        self._init_state = init_state
        self.state = init_state
        self.target=[0,0]
        self.velocity = [5, 5]
        self.max_velocity = velocity
        self.min_velocity = velocity/3
     #  self._update_state()

    def calculate_angle(self, x, y):
        if x >= 0:
            x += 0.0001
            phi = math.atan(y / x)
        else:
            phi = np.pi - math.atan(y / abs(x))
        return phi

    #创建传感器对象
    def create_sensor(self, num_rays=16, radius=40, min_angle=-45, max_angle=45,bounds=[10,10,10,10], obs=[]):
        self.sensor = RaySensor(num_rays,radius, min_angle, max_angle, bounds, obs)

    #运动积分
    def step(self, action):
        a_x = action[0] * self.a_max
        a_y = action[1] * self.a_max
        self.velocity[0] += a_x * self._dt
        self.velocity[1] += a_y * self._dt
        self.state[2] = self.calculate_angle(self.velocity[0], self.velocity[1])
        if  math.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2) >= self.max_velocity:
            self.velocity[0] = self.max_velocity * np.cos(self.state[2])
            self.velocity[1] = self.max_velocity * np.sin(self.state[2])

        if  math.sqrt(self.velocity[0] ** 2 + self.velocity[1] ** 2) <= self.min_velocity:
            self.velocity[0] = self.min_velocity * np.cos(self.state[2])
            self.velocity[1] = self.min_velocity * np.sin(self.state[2])

        self.state[0] += self.velocity[0] * self._dt
        self.state[1] += self.velocity[1] * self._dt


        #更新传感器
        # print(self.state)
        self.sensor.update(*self.state)


        # action为控制量，由外部控制器输入，表示航向角偏转刻度，可为连续量[-1 1]，也可为离散量[-1，0，1]
        #更新状态
        # self.state[2] += action * self._max_turnangle_perstep
        # # self.state[0] += self.velocity * np.cos(self.state[2]) * self._dt
        # # self.state[1] += self.velocity * np.sin(self.state[2]) * self._dt
        #
        # #更新传感器
        # self.sensor.update(*self.state)

        #更新绘图数据
     #   self._update_state()

    #重置状态
    def reset(self, new_state):
        self.state = self.generate_safe_position(new_state)
        self.sensor.update(*self.state)
     #  self._update_state()

    #给传感器传递真实信息(用于传感器计算碰撞和与障碍物距离)
    def set_sensor_locator(self, locator):
        self.sensor.set_locator(locator)

     #生成合法初始位置
    def generate_safe_position(self, state):
        #检验状态合法性
        self.sensor.update(state[0], state[1], state[2])
        if min(self.sensor.distances) >= 0.30:
            return state
        else: #不合法则随机生成一个状态
            while True:
                x, y = tuple(np.random.uniform(-100,100, 2))
                theta = np.random.uniform(0, 2 * np.pi)
                self.sensor.update(x,y,theta)
                if min(self.sensor.distances) >= 0.30:
                    return [x,y,theta]


    #将原始图标数据转换到当前位置，并旋转到当前方位（用于界面显示）
    def _update_state(self):
        # 经过旋转
        next_state = self.state
        t = vtk.vtkTransform()
        t.Translate([next_state[0], next_state[1], 0.])
        t.RotateZ(np.degrees(next_state[2]))
        self._polydata = filterUtils.transformPolyData(self._raw_polydata, t)


    def to_positioned_polydata(self):
        # 转换到当前位置的polydata
        return self._polydata

    def to_polydata(self):
        # 原始polydata
        return self._raw_polydata
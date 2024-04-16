# -*- coding: utf-8 -*=

import numpy as np
#import pandas as pd
#import director.src.python.director as dt

from director.debugVis import DebugData
from director import vtkAll as vtk
from director import ioUtils, filterUtils

"""世界模型"""


class World(object):
    # 初始化参数
    # 左上角 + +
    # 右上角 + -
    # 左下角 - +
    # 右下角 - -
    # 纵坐标 横坐标
    _obstacle_info = np.array(
        [[0, 370, 276, 0, 54, 5],
         [120, 12, 276, 0, 34, 5],
         [410, 70, 276, 0, 44, 5],
         [380, 237, 276, 0, 58, 5],
         [20, 170, 276, 0, 34, 5],
         [40, 230, 276, 0, 49, 5],
         [0, -440, 276, 0, 44, 5],
         [220, -17, 276, 0, 39, 5],
         [340, -180, 276, 0, 44, 5],
         [50, -270, 276, 0, 26, 5],
         [450, -420, 276, 0, 40, 5],
         [-320, 130, 276, 0, 50, 5],
         [-270, 400, 276, 0, 38, 5],
         [-230, 80, 276, 0, 28, 5],
         [-150, 20, 276, 0, 57, 5],
         [-450, 53, 276, 0, 25, 5],
         [-100, 420, 276, 0, 44, 5],
         [-150, -270, 276, 0, 20, 5],
         [-350, -70, 276, 0, 40, 5],
         [-250, -450, 276, 0, 20, 5],
         [-50, -70, 276, 0, 28, 5],

         ])


    # 初始化世界模型
    def __init__(self, width=300, height=300):
        self._obstacle_radius = [15, 25]
        self._obstacle_height = [10, 50]
        self._obstacle_velocity = [-30, 30]

        self._data = DebugData()
        self._width = width
        self._height = height
        self._add_boundaries()

        self.obstacle_num = 0
        self.obstacles = []

    # 绘制世界边界
    def _add_boundaries(self):
        self._x_max, self._x_min = self._height / 2, -self._height / 2
        self._y_max, self._y_min = self._width / 2, -self._width / 2

        corners = [
            (self._x_max, self._y_max, 0),  # Top-right corner.
            (self._x_max, self._y_min, 0),  # Bottom-right corner.
            (self._x_min, self._y_min, 0),  # Bottom-left corner.
            (self._x_min, self._y_max, 0)  # Top-left corner.
        ]

        # Loopback to begining.
        corners.append(corners[0])

        # draw the world aera
        for start, end in zip(corners, corners[1:]):
            self._data.addLine(start, end, radius=1)

    # 生成障碍物（按照比例随机生成）
    def generate_obstacles(self, random_obstacle=True, density=0.05, moving_obstacle_ratio=0.20):
        bounds = self._x_min, self._x_max, self._y_min, self._y_max
        if random_obstacle:
            field_area = self._width * self._height
            obstacle_area = int(field_area * density)
            while obstacle_area > 0:
                radius = np.random.uniform(self._obstacle_radius[0], self._obstacle_radius[1])  # 圆形障碍物的半径
                center_x_range = (self._x_min + radius, self._x_max - radius)
                center_y_range = (self._y_min + radius, self._y_max - radius)
                center_x = np.random.uniform(*center_x_range)
                center_y = np.random.uniform(*center_y_range)
                theta = np.random.uniform(0., 360.)  # 障碍物的速度方向
                obstacle_area -= np.pi * radius ** 2

                # Only some obstacles should be moving.
                if np.random.random_sample() >= moving_obstacle_ratio:
                    velocity = 0.0
                else:
                    velocity = np.random.uniform(self._obstacle_velocity[0], self._obstacle_velocity[1])  # 障碍物的速度大小

                height = np.random.uniform(self._obstacle_height[0], self._obstacle_height[1])  # 圆形障碍物的高度
                obs = Obstacle(center_x, center_y, np.radians(theta), velocity, radius, bounds, height)
                self.obstacles.append(obs)
                self.obstacle_num += 1

             #  print("[",int(center_x), int(center_y), int(theta), int(velocity), int(radius), int(height),"],")
        else:
            for centerX, centerY, theta, velocity, radius, height in zip(World._obstacle_info[:, 0],World._obstacle_info[:, 1],
                                                                         World._obstacle_info[:, 2],World._obstacle_info[:, 3],
                                                                         World._obstacle_info[:, 4],World._obstacle_info[:, 5]):
                obstacle = Obstacle(centerX, centerY, np.radians(theta), velocity, radius, bounds, height)
                self.obstacles.append(obstacle)
                self.obstacle_num += 1

    # 推进一个步长
    def step(self):
        for obstacle in self.obstacles:
            obstacle.step()

    # 绘图数据
    def to_polydata(self):
        """Converts world to visualizable poly data."""
        return self._data.getPolyData()


"""障碍物模型"""
class Obstacle(object):
    def __init__(self, center_x, center_y, yaw, velocity, radius, bounds, height, color=[1, 1, 1]):
        self.velocity = 0
        self._dt = 0.1
        self._data = DebugData()
        self.state = np.array([0,0,0])  #初始化
        self.state[0] = center_x
        self.state[1] = center_y
        self.state[2] = yaw
        self.velocity = float(velocity)
        self.radius = radius
        self.height = height
        self.bounds = bounds
        center = [0, 0, height/2-1]
        axis = [0, 0, 1]  # Upright cylinder.
        self._data.addCylinder(center, axis, height, radius, color)  # 圆柱体，可以改为其他形状

        #绘图用
        self._raw_polydata = self._data.getPolyData()
        self._update_state()

    def step(self):
        self.state[2] = self._control()
        self.state[0] += self.velocity * np.cos(self.state[2]) * self._dt
        self.state[1] += self.velocity * np.sin(self.state[2]) * self._dt

        # 坐标转换到显示坐标系
        if self.velocity:
            self._update_state()

    def reset(self, new_state):
        self.state = new_state

        # 坐标转换到显示坐标系
        if self.velocity:
            self._update_state()

    #计算控制量（航向角）
    def _control(self):
        x_min, x_max, y_min, y_max = self.bounds
        x, y, theta = self.state
        if x - self.radius <= x_min:
            return np.pi
        elif x + self.radius >= x_max:
            return np.pi
        elif y - self.radius <= y_min:
            return np.pi
        elif y + self.radius >= y_max:
            return np.pi
        else:
            return theta

    #将原始图标数据转换到当前位置，并旋转到当前方位（用于界面显示）
    def _update_state(self):
        #经过旋转
        next_state = self.state
        t = vtk.vtkTransform()
        t.Translate([next_state[0], next_state[1], 0.])
        t.RotateZ(np.degrees(next_state[2]))
        self._polydata = filterUtils.transformPolyData(self._raw_polydata, t)
        # self._state = next_state

    def to_positioned_polydata(self):
        #经过旋转以后的polydata
        return self._polydata

    def to_polydata(self):
        #原始polidata
        return self._raw_polydata

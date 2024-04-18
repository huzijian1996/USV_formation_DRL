# -*- coding: utf-8 -*-

import numpy as np
import math
from director import vtkAll as vtk
from director import ioUtils, filterUtils
from sensor import RaySensor

np.random.seed(1)
from director.debugVis import DebugData
from director import vtkAll as vtk
from director import ioUtils, filterUtils


class Sea(object):
    # 创建一个Agent
    def __init__(self, state=[145, 145, 0]):
        # 初始化
        self.state = state

        model = "./obj/sea.obj"

        # 显示转换
        t = vtk.vtkTransform()
        t.Scale(300, 200, 300)  # 模型大小
        polydata = ioUtils.readPolyData(model)
        self._raw_polydata = filterUtils.transformPolyData(polydata, t)

    # 将原始图标数据转换到当前位置，并旋转到当前方位（用于界面显示）
    def _update_state(self):
        t = vtk.vtkTransform()
        t.Translate([self.state[0], self.state[1], 0.])
        t.RotateX(np.degrees(np.pi/2))
        self._polydata = filterUtils.transformPolyData(self._raw_polydata, t)

    def to_positioned_polydata(self):
        # 转换到当前位置的polydata
        return self._polydata

    def to_polydata(self):
        # 原始polydata
        return self._raw_polydata


class Island(object):
    #创建一个Agent
    def __init__(self, ID = 0, state=[145, 145, 0]):
        #初始化
        self.state = state

        t = vtk.vtkTransform()
        if ID == 0:
            model = "./obj/island0.OBJ"
            self.state[2] += 50
            t.Scale(80, 80, 80) #模型大小
        elif ID == 1:
            model = "./obj/island1.OBJ"
            self.state[1] -= 12
            self.state[2] -= 30
            t.Scale(1.2, 1.2, 1.2) #模型大小
        elif ID == 2:
            model = "./obj/island2.OBJ"
            self.state[1] += 15
            self.state[2] -= 95
            t.Scale(1.3, 1, 1) #模型大小
        elif ID == 3:
            model = "./obj/island3.OBJ"
            self.state[1] -= 7
            self.state[2] -= 30
            t.Scale(6, 6, 6)  # 模型大小
        elif ID == 4:
            model = "./obj/island4.OBJ"
            self.state[0] -= 8
            self.state[1] -= 5
            self.state[2] -= 65
            t.Scale(12, 12, 12)  # 模型大小
        elif ID == 5:
            model = "./obj/island5.OBJ"
            self.state[0] += 28
            self.state[1] -= 23
            self.state[2] -= 150
            t.Scale(5, 5, 6.2)  # 模型大小
        elif ID == 6:
            model = "./obj/island6.OBJ"
            self.state[0] += 15
            self.state[1] -= 12
            self.state[2] -= 15
            t.Scale(2, 1, 2)  # 模型大小
        elif ID == 7:
            model = "./obj/island7.OBJ"
            self.state[0] += 6
            self.state[1] += 35
            self.state[2] -= 20
            t.Scale(2, 3, 2)  # 模型大小
        elif ID == 8:
            model = "./obj/island8.OBJ"
            self.state[0] += 13
            self.state[1] -= 25
            self.state[2] -= 45
            t.Scale(10, 15, 10)  # 模型大小
        elif ID == 9:
            model = "./obj/island9.OBJ"
            self.state[0] += 6
            self.state[1] += 3
            self.state[2] -= 20
            t.Scale(5, 5, 5)  # 模型大小
        elif ID == 10:
            model = "./obj/island10.OBJ"
            self.state[0] += 18
            self.state[1] -= 8
            self.state[2] -= 45
            t.Scale(6, 5, 6)  # 模型大小
        elif ID == 11:
            model = "./obj/island6.OBJ"
            self.state[0] += 28
            self.state[1] -= 20
            self.state[2] -= 15
            t.Scale(2, 1, 2)  # 模型大小
        elif ID == 12:
            model = "./obj/island11.OBJ"
            # self.state[0] += 28
            self.state[1] += 120
            self.state[2] -= 60
            t.Scale(1.2, 1, 1)  # 模型大小
        elif ID == 13:
            model = "./obj/island3.OBJ"
            # self.state[0] += 8
            self.state[1] -= 20
            self.state[2] -= 47
            t.Scale(10, 10, 10)
        polydata = ioUtils.readPolyData(model)
        self._raw_polydata = filterUtils.transformPolyData(polydata, t)


    #将原始图标数据转换到当前位置，并旋转到当前方位（用于界面显示）
    def _update_state(self):
        t = vtk.vtkTransform()
        t.Translate([self.state[0], self.state[1], 0.])
        # t.RotateZ(np.degrees(self.state[2]))
        self._polydata = filterUtils.transformPolyData(self._raw_polydata, t)


    def to_positioned_polydata(self):
        # 转换到当前位置的polydata
        return self._polydata

    def to_polydata(self):
        # 原始polydata
        return self._raw_polydata
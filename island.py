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

        model = "sea.obj"

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
    def __init__(self, state=[145, 145, 0]):
        #初始化
        self.state = state

        model = "Rock.obj"

        #显示转换
        t = vtk.vtkTransform()
        t.Scale(0.3, 0.3, 0.3) #模型大小
        polydata = ioUtils.readPolyData(model)
        self._raw_polydata = filterUtils.transformPolyData(polydata, t)

    #将原始图标数据转换到当前位置，并旋转到当前方位（用于界面显示）
    def _update_state(self):
        t = vtk.vtkTransform()
        t.Translate([self.state[0], self.state[1], 0.])
        t.RotateZ(np.degrees(self.state[2]))
        self._polydata = filterUtils.transformPolyData(self._raw_polydata, t)


    def to_positioned_polydata(self):
        # 转换到当前位置的polydata
        return self._polydata

    def to_polydata(self):
        # 原始polydata
        return self._raw_polydata
# -*- coding: utf-8 -*=

import math
import numpy as np
import director.vtkAll as vtk
from director.debugVis import DebugData


#激光雷达传感器
class RaySensor(object):

    def __init__(self, num_rays=16, radius=40, min_angle=-45, max_angle=45, bounds=[10,10,10,10], world_obstacles=[]):
        """Constructs a RaySensor.

        Args:
            num_rays: Number of rays.
            radius: Max distance of the rays.
            min_angle: Minimum angle of the rays in degrees.
            max_angle: Maximum angle of the rays in degrees.
        """
        self._num_rays = num_rays
        self._radius = radius
        self._min_angle = math.radians(min_angle)  #degree to rad
        self._max_angle = math.radians(max_angle)

        self._locator = None
        self._state = [0., 0., 0.]  # x, y, theta  position and heading 

        self._hit = np.zeros(num_rays) # wether the rays detect a object
        self._distances = np.zeros(num_rays) #the dis between the object and  sensor
        self._intersections = [[0, 0, 0] for i in range(num_rays)]

        self._update_rays(self._state[2])
        self.bounds = bounds
        self.obstacles = world_obstacles

    @property
    def distances(self):
        """Array of distances measured by each ray."""
        normalized_distances = [
            # self._distances[i] / self._radius if self._hit[i] else 1.0
            self._distances[i]  if self._hit[i] else 1.0
        for i in range(self._num_rays)
        ]
        return normalized_distances

    def has_collided(self, max_distance=0.05):
        """Returns whether a collision has occured or not.

        Args:
            max_distance: Threshold for collision distance.
        """

        for i in range(len(self.obstacles)):
            if (self._state[0]-self.obstacles[i].state[0])**2 + (self._state[1]-self.obstacles[i].state[1])**2 <= self.obstacles[i].radius**2:
                return True
        if (self._state[0] <= self.bounds[0]) or (self._state[0] >= self.bounds[1])or(self._state[1] <= self.bounds[2]) or (self._state[1] >= self.bounds[3]):
            return True



        return False

    def set_locator(self, locator):
        """Sets the vtk cell locator.

        Args:
            locator: Cell locator.
        """
        self._locator = locator

    def update(self, x, y, theta):
        """Updates the sensor's readings.

        Args:
            x: X coordinate.
            y: Y coordinate.
            theta: Yaw.
        """
        self._update_rays(theta)
        origin = np.array([x, y, 0])
        self._state = [x, y, theta]

        if self._locator is None:
            return

        for i in range(self._num_rays):
            hit, dist, inter = self._cast_ray(origin, origin + self._rays[i])
            self._hit[i] = hit
            self._distances[i] = dist
            self._intersections[i] = inter

    def _update_rays(self, theta):
        """Updates the rays' readings.

        Args:
            theta: Yaw.
        """
        r = self._radius
        angle_step = (self._max_angle - self._min_angle) / (self._num_rays - 1)
        self._rays = [
            np.array([
                r * math.cos(theta + self._min_angle + i * angle_step),
                r * math.sin(theta + self._min_angle + i * angle_step),
                0
            ])
            for i in range(self._num_rays)
        ]

    def _cast_ray(self, start, end):
        """Casts a ray and determines intersections and distances.

        Args:
            start: Origin of the ray.
            end: End point of the ray.

        Returns:
            Tuple of (whether it intersected, distance, intersection).
        """
        tolerance = 0.0                 # intersection tolerance
        pt = [0.0, 0.0, 0.0]            # coordinate of intersection
        distance = vtk.mutable(0.0)     # distance of intersection
        pcoords = [0.0, 0.0, 0.0]       # location within intersected cell
        subID = vtk.mutable(0)          # subID of intersected cell

        hit = self._locator.IntersectWithLine(start, end, tolerance,
                                              distance, pt, pcoords, subID)

        return hit, distance, pt

    def to_polydata(self):
        """Converts the sensor to polydata."""
        d = DebugData()
        origin = np.array([self._state[0], self._state[1], 0])
        for hit, intersection, ray in zip(self._hit,
                                          self._intersections,
                                          self._rays):
            if hit:
                color = [1., 0.45882353, 0.51372549]
                endpoint = intersection
            else:
                color = [0., 0.6, 0.58823529]
                endpoint = origin + ray

            d.addLine(origin, endpoint, color=color, radius=0.4) #line style

        return d.getPolyData()

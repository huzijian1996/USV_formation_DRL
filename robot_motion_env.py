# -*- coding: utf-8 -*-
# alias director=/Users/wankaifang/director/build/install/bin/directorPython
# cd /Users/wankaifang/build/RL-Robot-Motion-planning/Env

import sys
import numpy as np
import math
from world import World
from world import Obstacle
from robot import Robot
from PythonQt import QtGui
from director import applogic
from director import vtkAll as vtk
from director import objectmodel as om
from director.debugVis import DebugData
from director import visualization as vis
from director.consoleapp import ConsoleApp
from director.timercallback import TimerCallback
import matplotlib.pyplot as plt
from collections import deque
from MADDPG.replay_buffer import ReplayBuffer

import time
import threading


np.random.seed(1)


PI = 3.141592653
DELTA = 0.01

"""
————————————————————————奖励值设定建议——————————————————————————
多数奖励在（-2,2）变化时 最优最差奖励设为3,-3 要比设为100,-100收敛的快 
因此是不是可能存在类似的大数吃小数问题
"""


"""目标运动控制环境"""
class RobotMotionEnv(object):
    def __init__(self, envname="USV",discrete_action=True):
        #初始化信息
        # 定义初始化参数
        self._world_width = 1000.0
        self._world_height = 1000.0
        self._random_obstacle = False
        self._obstacle_density = 0.15
        self._moving_obstacle_ratio = 0.0


        self._robot_init_velocity = 25
        self._sensor_num_rays = 32
        self._sensor_radius = 100
        self._sensor_detect_scope = [-90, 90]
        self._target_init_state = [-400, -400, 0]
        self._target_init_velocity = 0
        self._random_pos = True
        self._world_bounds = -self._world_height / 2,\
                             self._world_height / 2,\
                             -self._world_width / 2,\
                             self._world_width / 2


        self._discrete_action = discrete_action
        self.action_dim = 2
        self.state_dim = self._sensor_num_rays + 3  # 传感器的16个测量值+d_x+d_y+d_theta

        self._locator = None
        self.ctrl = []



        # 是否结束标志
        self._all_hit_target = False  # 命中
        self._all_collided = False  # 碰撞
        self._terminal = False  # 结束
        self._hit_target = False  # 命中
        self._collided = False  # 碰撞
        self._timeout = False

        # 是否需要持续更新locator
        self._locator_continu_update = False if self._moving_obstacle_ratio <= 0 else True

        # 记录命中/碰撞次数
        self._num_targets = 0
        self._num_crashes = 0
        self._num_timeout = 0
        self._episode_iter = 0
        self._max_episode = 10
        self._predict_iter = 0
        self._max_predict = 100
        self._all_iter = 0
        self.hit_probability = []  #成功命中目标的概率_num_targets/_max_episode

        self.is_hit_target = deque(maxlen=100) #用来记录是否命中目标 命中则为1否则则为0
        self.hit_rate = []

        self.formation_dis = 20
        self.robot_num = 3
        self.robots = []
        self.robot_init_states= [[300, 100, 0],[-100, -300, 0],[0, -10, 0]]
        """创建并初始化对象"""
        #世界对象（包括障碍物）
        self._world = World(self._world_width, self._world_height)
        self._world.generate_obstacles(self._random_obstacle, self._obstacle_density, self._moving_obstacle_ratio)

        #robot对象
        self.obs_num = len(self._world.obstacles)
        for i in range (self.robot_num):
            robot = Robot(envname,self.robot_init_states[i], self._robot_init_velocity)
            robot.create_sensor(self._sensor_num_rays, self._sensor_radius, self._sensor_detect_scope[0],
                                self._sensor_detect_scope[1],np.array(self._world_bounds),self._world.obstacles)
            self._update_locator()  # 更新locator
            robot.set_sensor_locator(self._locator)
            self.robots.append(robot)



        #target对象(用一个不同颜色的障碍物代替)
        self._target = Obstacle(self._target_init_state[0], self._target_init_state[1], self._target_init_state[2], self._target_init_velocity, 20, self._world_bounds, 2, color=[0, 0.8, 0])

        #目标绑定target
        self.update_robot_target()

        #行动
        self._discrete_action = discrete_action
        if discrete_action:
            self.actions = [-1,0,1]    #离散行动空间，全力左转/全力前进/全力右转
        else:
            self.action_bound = [-1,1] #连续行动空间，控制量在【-1 1】间

        #初始观测
        self.observation = []
        self.observation_ = []
        for i in range(self.robot_num):
            obs = self._get_env_state(i)
            self.observation.append(obs)
            self.observation_.append(obs)

        #创建窗体
        self._view = None
        self._app = None
        self._frame_target = None
        self._frame_robot = []
        self._frame_obstacles = []
        self.create_view()

    #单步推进

    def world_step(self):
        self._world.step()  #世界推进
        self._target.step() #目标推进

        self.update_robot_target() #目标绑定

        self._timeout = True if self._predict_iter == self._max_predict else False
        if self._all_hit_target:
            self._num_targets += 1
            self.is_hit_target.append(1)
        if self._all_collided:
            self._num_crashes += 1
            self.is_hit_target.append(0)
        if self._timeout:
            self._num_timeout += 1
            self.is_hit_target.append(0)
        # print('0state',self.robots[0].state)
        # for i in range(self.robot_num):
        #     print(i,self.robots[i].target)


    def robot_step(self, i, action):
        self._hit_target = False
        self._collided = False
        #解析action

        # if self._discrete_action:
        #     action = self.actions[action]
        # else:
        #     action = np.clip(action, *self.action_bound)[0]


        #对象推进
        prev_relative_pos = [self.robots[i].target[0] - self.robots[i].state[0], self.robots[i].target[1] - self.robots[i].state[1]]

        if self._locator_continu_update:
            self._update_locator() #更新locator
            self.robots[i].set_sensor_locator(self._locator)

        self.robots[i].step(action)

        self.update_robot_target()




        #更新结束标志
        # 更新结束标志
        self._hit_target = self._is_hit_target(i=i)
        if self._hit_target and i == 0:
            self._all_hit_target = True

        if self.robots[i].sensor.has_collided():
            self._collided = True
            self._all_collided = True

        #获取下一步状态和收益
        new_env_state = self._get_env_state(i)

        # new_relative_pos = [self._target.state[0] - self.robots[i].state[0], self._target.state[1] - self.robots[i].state[1]]
        new_relative_pos = [self.robots[i].target[0] - self.robots[i].state[0], self.robots[i].target[1] - self.robots[i].state[1]]

        if i == 0:
            reward = self._get_reward_leader(prev_relative_pos, new_relative_pos,i)
        else:
            reward = self._get_reward_follower(prev_relative_pos, new_relative_pos, i)

        terminal = (self._hit_target and i==0) or self._collided or self._timeout

        return new_env_state, reward, terminal




    #程序重置
    def reset(self):
        self._terminal = False

        self._all_hit_target = False  # 命中
        self._all_collided = False  # 碰撞
        self._hit_target = False
        self._collided = False
        self._timeout = False
        self._predict_iter = 0


        if self._random_pos:  #随机产生位置
            for i in range(self.robot_num):
                self.robots[i].reset(self._generate_safe_position_robot(id=i))
            self._target.reset(self._generate_safe_position_target())

        self.update_robot_target()
        env_state = []
        for i in range(self.robot_num):
            env_state.append(self._get_env_state(i))

        return env_state

    #界面刷新
    def render(self):
        self.update_view()
        self._view.update()



    #行动采样（随机）
    def sample_action(self):
        if self._discrete_action:
            a = np.random.choice(list(range(len(self.actions))))
        else:
            a = np.random.uniform(*self.action_bound, size=self.action_dim)
        return a


    def update_robot_target(self):
        for i in range(self.robot_num):
            if i == 0:
                self.robots[i].target[0] = self._target.state[0]
                self.robots[i].target[1] = self._target.state[1]
            elif i == 1:
                self.robots[i].target[0], self.robots[i].target[1] = self.coordinate_conversion(-self.formation_dis,
                                                                                self.formation_dis,
                                                                                self.robots[0].state[0],
                                                                                self.robots[0].state[1],
                                                                                self.robots[0].state[2])
            else:
                self.robots[i].target[0], self.robots[i].target[1] = self.coordinate_conversion(-self.formation_dis,
                                                                                -self.formation_dis,
                                                                                self.robots[0].state[0],
                                                                                self.robots[0].state[1],
                                                                                self.robots[0].state[2])


    def _generate_safe_position_robot(self, id=0):
        if id==0:
            while True:
                pos = [np.random.uniform(-self._world_height/2+50, self._world_height/2-50),np.random.uniform(-self._world_width/2+50, self._world_width/2-50),0]
                count = 0
                for i in range(len(self._world.obstacles)):
                    dx = (pos[0] - self._world.obstacles[i].state[0]) ** 2
                    dy = (pos[1] - self._world.obstacles[i].state[1]) ** 2
                    dz = (pos[2] - self._world.obstacles[i].state[2]) ** 2
                    d = math.sqrt(dx + dy + dz)
                    if d > (self._world.obstacles[i].radius + 50):
                        count += 1
                if count == self.obs_num:
                    break
        else:
            while True:
                pos = [np.random.uniform(-self._world_height/2+50, self._world_height/2-50),np.random.uniform(-self._world_width/2+50, self._world_width/2-50),0]
                if math.sqrt((pos[0]-self.robots[0].state[0])**2 + (pos[1]-self.robots[0].state[1])**2) <= 100:
                    count = 0
                    for i in range(len(self._world.obstacles)):
                        dx = (pos[0] - self._world.obstacles[i].state[0]) ** 2
                        dy = (pos[1] - self._world.obstacles[i].state[1]) ** 2
                        dz = (pos[2] - self._world.obstacles[i].state[2]) ** 2
                        d = math.sqrt(dx + dy + dz)
                        if d > (self._world.obstacles[i].radius + 50):
                            count += 1
                    if count == self.obs_num:
                        break

        return pos

    def _generate_safe_position_target(self):
        while True:
            pos = [np.random.uniform(-self._world_height/2+50, self._world_height/2-50),np.random.uniform(-self._world_width/2+50, self._world_width/2-50),0]
            count = 0
            for i in range(len(self._world.obstacles)):
                dx = (pos[0] - self._world.obstacles[i].state[0]) ** 2
                dy = (pos[1] - self._world.obstacles[i].state[1]) ** 2
                dz = (pos[2] - self._world.obstacles[i].state[2]) ** 2
                d = math.sqrt(dx + dy + dz)
                if d >= (self._target.radius + self._world.obstacles[i].radius):
                    count += 1
            if count == self.obs_num:
                break
        return pos

    #坐标系转换
    def coordinate_conversion(self, x1, y1, x0, y0, theta):
        x = x1 * np.cos(theta) - y1 * np.sin(theta) + x0
        y = x1 * np.sin(theta) + y1 * np.cos(theta) + y0
        return x ,y

    #是否命中目标
    def _is_hit_target(self, threshold=20, i=0):
        if (abs(self.robots[i].state[0] - self._target.state[0]) <= threshold and
            abs(self.robots[i].state[1] - self._target.state[1]) <= threshold):
            return True


    #收益计算（reward设计很重要）
    def _get_reward_leader(self, prev_ralative_pos, new_ralative_pos, i):
        prev_distance = np.sqrt(prev_ralative_pos[0] ** 2 + prev_ralative_pos[1] ** 2)
        new_distance = np.sqrt(new_ralative_pos[0] ** 2 + new_ralative_pos[1] ** 2)
        if self._collided:
            return -10
        elif self._hit_target:
            return 5
        else:
            delta_distance = prev_distance - new_distance  # 敌我距离增量
            angle_distance = -abs(self._angle_to_destination(new_ralative_pos[0],new_ralative_pos[1],i)) / 4   # 角度偏移量 <0

            total = 0
            ray_k = int(self._sensor_num_rays/5)
            for ray in range(ray_k):
                total += (self.robots[i].sensor.distances[2*ray_k + ray] - 1)
            obstacle_ahead = total / ray_k #插的越深 负值越多

            # obstacle_ahead = self.robots[i].sensor.distances[int(self._sensor_num_rays / 2)] - 1

            # total = 0
            # for ray in range(self._sensor_num_rays):
            #     total += (self.robots[i].sensor.distances[ray] - 1)
            # obstacle_ahead = total / self._sensor_num_rays#插的越深 负值越多

            lamda1, lamda2, lamda3 = 0.4, 0.2, 0.4
            return lamda1*delta_distance + lamda2*angle_distance + lamda3*obstacle_ahead

    def _get_reward_follower(self, prev_ralative_pos, new_ralative_pos, i):
        prev_distance = np.sqrt(prev_ralative_pos[0] ** 2 + prev_ralative_pos[1] ** 2)
        new_distance = np.sqrt(new_ralative_pos[0] ** 2 + new_ralative_pos[1] ** 2)
        if self._collided:
            return -10
        elif new_distance <= 5:
            return 3*(1-new_distance/5)
        else:
            delta_distance = prev_distance - new_distance  # 敌我距离增量
            angle_distance = -abs(self._angle_to_destination(new_ralative_pos[0],new_ralative_pos[1],i)) / 4  # 角度偏移量 <0

            total = 0
            ray_k = int(self._sensor_num_rays/5)
            for ray in range(ray_k):
                total += (self.robots[i].sensor.distances[2*ray_k + ray] - 1)
            obstacle_ahead = total / ray_k #插的越深 负值越多

            # obstacle_ahead = self.robots[i].sensor.distances[int(self._sensor_num_rays / 2)] - 1

            # total = 0
            # for ray in range(self._sensor_num_rays):
            #     total += (self.robots[i].sensor.distances[ray] - 1)
            # obstacle_ahead = total / self._sensor_num_rays  # 插的越深 负值越多

            lamda1, lamda2, lamda3 = 0.4, 0.2, 0.4
            return lamda1 * delta_distance + lamda2 * angle_distance + lamda3 * obstacle_ahead


     #返回环境状态
    def _get_env_state(self,i):
        dx = self.robots[i].target[0]-self.robots[i].state[0]
        dy = self.robots[i].target[1]-self.robots[i].state[1]
        relative_state = [dx / self._world_height, dy / self._world_width, self._angle_to_destination(dx,dy,i) / np.pi]
        dis = [x - 1 for x in self.robots[i].sensor.distances]
        return np.hstack([relative_state, dis])
        # return np.hstack([relative_state, self.robots[i].sensor.distances])

    #计算偏航角与目标线的差
    def _angle_to_destination(self,x,y,i):
        return self._wrap_angles(np.arctan2(y, x) - self.robots[i].state[2])

    #角度转换
    def _wrap_angles(self, a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    #更新环境信息，传递给传感器
    def _update_locator(self):
        #判断是否需要更新
        d = DebugData()
        d.addPolyData(self._world.to_polydata())
        for obstacle in self._world.obstacles:
            d.addPolyData(obstacle.to_positioned_polydata())
        self._locator = vtk.vtkCellLocator()
        self ._locator.SetDataSet(d.getPolyData())
        self._locator.BuildLocator()

    #画图专用
    def _add_polydata(self, polydata, frame_name, color):
        om.removeFromObjectModel(om.findObjectByName(frame_name))
        frame = vis.showPolyData(polydata, frame_name, color=color)

        vis.addChildFrame(frame)
        return frame

    #动目标坐标转换
    def _update_moving_object_obs(self, moving_object_state, frame):
        t = vtk.vtkTransform()
        t.Translate(moving_object_state[0], moving_object_state[1], 0.0)
        t.RotateZ(np.degrees(moving_object_state[2]))
        frame.getChildFrame().copyFrame(t)

    def _update_moving_object_robot(self, moving_object_state, frame):
        t = vtk.vtkTransform()
        t.Translate(moving_object_state[0], moving_object_state[1], 0.0)
        t.RotateZ(np.degrees(moving_object_state[2]))
        t.RotateY(np.degrees(np.pi / 2))
        t.RotateX(np.degrees(np.pi))
        t.RotateZ(np.degrees(np.pi / 2))
        frame.getChildFrame().copyFrame(t)

    #更新传感器绘图数据
    def _update_sensor(self, sensor, frame_name):

        vis.updatePolyData(sensor.to_polydata(), frame_name,
                           colorByName="RGB255")
    #创建视图
    def create_view(self):
        # 创建窗体
        self._app = ConsoleApp()
        self._view = self._app.createView(useGrid=False)
        # 绑定窗体和world
        om.removeFromObjectModel(om.findObjectByName("world"))
        vis.showPolyData(self._world.to_polydata(), "world")
        # 绑定目标
        self._frame_target = self._add_polydata(self._target.to_polydata(), "target", [0, 0.8, 0])
        self._update_moving_object_obs(self._target.state, self._frame_target)
        # 绑定robot

        for i,robot in enumerate(self.robots):
            frame_name = "robot{}".format(i + 1)
            if i == 0:
                frame_robot =  self._add_polydata(robot.to_polydata(), frame_name, [0.8, 0.2, 0.6])
            else:
                frame_robot =  self._add_polydata(robot.to_polydata(), frame_name, [0.4, 0.2, 0.9])

            self._update_moving_object_robot(robot.state, frame_robot)
            self._frame_robot.append(frame_robot)
        # 绑定obstacles
        for i, obs in enumerate(self._world.obstacles):
            frame_name = "obstacle{}".format(i+1)
            frame_obstacle = self._add_polydata(obs.to_polydata(), frame_name, [1.0, 1.0, 1.0])
            self._update_moving_object_obs(obs.state, frame_obstacle)
            self._frame_obstacles.append(frame_obstacle)

    #更新视图
    def update_view(self):
        # 更新运动对象
        self._update_moving_object_obs(self._target.state, self._frame_target)
        for i,robot in enumerate(self.robots):
            self._update_moving_object_robot(robot.state, self._frame_robot[i])

        for i, robot in enumerate(self.robots):
            frame_name = "rays{}".format(i + 1)
            self._update_sensor(robot.sensor, frame_name)

        for i, obstacle in enumerate(self._world.obstacles):
            self._update_moving_object_obs(obstacle.state, self._frame_obstacles[i])




    #结果输出
    def plot_out(self):
        #控制器画图
        # if (self.ctrl is not None) and (len(self.ctrl.cost_his)) > 0:
        #     fig0 = plt.figure("train cost")
        #     plt.plot(np.arange(len(self.ctrl.cost_his)), self.ctrl.cost_his)
        #     plt.ylabel('Cost')
        #     plt.xlabel('training steps')
        #     fig0.show()


        fig_hr = plt.figure("Hit Rate")
        plt.plot(np.arange(len(self.hit_rate)), self.hit_rate)
        plt.ylabel('Hit Rate')
        plt.xlabel('Episodes')
        fig_hr.savefig('./MADDPG/pic/Hit Rate_maddpg.png')

        for i in range(self.robot_num):
            plt.clf()
            fig = plt.figure("Average Reward")
            plt.plot(np.arange(len(self.robots[i].episode_average_reward)), self.robots[i].episode_average_reward)
            plt.ylabel('Episode Average Reward')
            plt.xlabel('Episodes')
            fig.savefig(f'./MADDPG/pic/average_reward_maddpg_{i}.png')
        for i in range(self.robot_num):
            plt.clf()
            fig = plt.figure("Actor Loss")
            plt.plot(np.arange(len(self.ctrl[i].a_loss)), self.ctrl[i].a_loss)
            plt.ylabel('Actor Loss')
            plt.xlabel('Tarining Steps')
            fig.savefig(f'./MADDPG/pic/Actor Loss_maddpg_{i}.png')
        for i in range(self.robot_num):
            plt.clf()
            fig = plt.figure("Critic Loss")
            plt.plot(np.arange(len(self.ctrl[i].c_loss)), self.ctrl[i].c_loss)
            plt.ylabel('Critic Loss')
            plt.xlabel('Tarining Steps')
            fig.savefig(f'./MADDPG/pic/Critic Loss_maddpg_{i}.png')


    def save_data(self):
        m = np.array(self.hit_rate)

        #保存命中率
        np.save('./MADDPG/data/hit_rate_maddpg_all',m)
        #保存回合平均奖励值
        for i in range(self.robot_num):
            r = np.array(self.robots[i].episode_average_reward)
            np.save(f'./MADDPG/data/average_reward_maddpg_{i}', r)



    def after_mainloop(self, display=True, mode="learn",episode=100,predict=1000, ctrl=None,timer=120,callback=None,args=None):
        #窗体显示与否
        self.args = args
        self.actions_num = args.N
        self.replay_buffer=ReplayBuffer(self.args)
        self.noise_std = self.args.noise_std_init
        self._display = display
        if display:
            widget = QtGui.QWidget()
            layout = QtGui.QVBoxLayout(widget)
            layout.addWidget(self._view)
            widget.showMaximized()
            applogic.resetCamera(viewDirection=[0.2, 0, -1])  #初始视场角

        # 学习或测试参数
        self._max_episode = episode
        self._max_predict = predict


        self.ctrl=ctrl

        #学习或测试
        if mode == "train":
            self._is_learn = True
        elif mode == "test":
            self._is_learn = False
        else:
            self._is_learn = True

        #学习模式
        if self._is_learn:
            # 开启学习计时器
            self.learn_timer = TimerCallback(targetFps=timer)
            self.learn_timer.callback = self.train
            self.learn_timer.start()

        else:
            for i in range (self.robot_num):
                self.ctrl[i].actor = self.ctrl[i].load_model(i)  #加载参数
            self.test_timer = TimerCallback(targetFps=timer)
            self.test_timer.callback = self.test
            self.test_timer.start()
            # self.test_timer.singleShot(1)

        #启动程序
        self._app.start()


    #————————————————学习主函数——————————————————————
    def train(self):
        #全部回合结束
        if self._episode_iter == self._max_episode:
            for i in range (self.robot_num):
                self.ctrl[i].save_model(i) # 保存参数
            self.plot_out()   # 绘图
            self.save_data() #保存输出结果
            print('learning process over!')
            self.learn_timer.stop()
            sys.exit()#结束程序
        #一个回合结束
        elif self._terminal:
            self._episode_iter += 1
            for i in range(self.robot_num):
                ave_r = sum(self.robots[i].episode_step_reward)/len(self.robots[i].episode_step_reward)
                self.robots[i].episode_average_reward.append(ave_r)

            print('episode',self._episode_iter,'step', self._predict_iter,'number of crashes',self._num_crashes,'number of targets',self._num_targets)

            # 20个回合保存一次参数
            if self._episode_iter % 20 == 0:
                for i in range(self.robot_num):
                    self.ctrl[i].save_model(i) # 保存参数

            self.observation = self.reset()
            self.observation_ = self.reset()




            if self._episode_iter >= 1:
                count = 0.0
                for i in range(len(self.is_hit_target)):
                    if self.is_hit_target[i] == 1:
                        count += 1
                count /= 200
                self.hit_rate.append(count)
                print('hit_rate', count)
        #——————————————————————————————MADDPG————————————————————————————————
        else:
            #计数增加
            self._predict_iter += 1
            #刷新界面
            if self._display:
                self.render()
            all_action = []
            all_done = []
            all_reward = []
            for i in range (self.robot_num):
                # 选择行动
                observation = self.observation[i].copy()

                action = self.ctrl[i].choose_action(observation, noise_std=self.noise_std)
                observation_, reward, done = self.robot_step(i, action)#单个智能体推进
                all_action.append(action)
                all_reward.append(reward)
                if done:
                    all_done.append(1)
                else:
                    all_done.append(0)
                self.observation_[i] = observation_.copy()
                self.robots[i].episode_step_reward.append(reward)



            self.replay_buffer.store_transition(self.observation,all_action,all_reward,self.observation_,all_done)
            self.observation = self.observation_.copy()


            if self.args.use_noise_decay:
                self.noise_std = self.noise_std - self.args.noise_std_decay if self.noise_std - self.args.noise_std_decay > self.args.noise_std_min else self.args.noise_std_min

            if self.replay_buffer.current_size > self.args.batch_size:
                # Train each agent individually
                for i in range(self.robot_num):
                    self.ctrl[i].train(self.replay_buffer, self.ctrl)

            if sum(all_done) != 0:
                self._terminal = True
            self.world_step()#世界推进
            #总迭代次数
            self._all_iter += 1





    #————————————————————————测试主函数——————————————————————————
    def test(self):
        #全部回合结束
        if self._episode_iter == self._max_episode:
            self.plot_out()  #绘图
            # self.save_date() #保存数据
            print('test process over!')
            self.test_timer.stop()
        #一个回合结束
        elif self._terminal:
            self._episode_iter += 1
            self.hit_probability.append(float(self._num_targets)/float(self._max_episode))
            print(self._episode_iter, self._predict_iter, self._num_crashes, self._num_targets, self.hit_probability[-1])

            self.observation = self.reset()
        else:
            # 计数增加
            self._predict_iter += 1
            # 刷新界面
            if self._display:
                self.render()
            all_done = []

            for i in range(self.robot_num):
                # 选择行动
                observation = self.observation[i].copy()

                action = self.ctrl[i].choose_action(observation, noise_std=0)
                observation_, reward, done = self.robot_step(i, action)  # 单个智能体推进
                if done:
                    all_done.append(1)
                else:
                    all_done.append(0)
                self.observation_[i] = observation_.copy()

            self.observation = self.observation_.copy()

            if sum(all_done) != 0:
                self._terminal = True
            self.world_step()  # 世界推进
            # 总迭代次数
            self._all_iter += 1
#test
if __name__ == "__main__":
    np.random.seed(10)
    env = RobotMotionEnv("UAV-0", False)
    env.after_mainloop(display=True, mode="test",episode=20,predict=100, ctrl=None)






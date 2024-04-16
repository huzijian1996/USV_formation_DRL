# -*- coding: utf-8 -*-
# cd /Users/wankaifang/build/RL-Robot-Motion-planning/

from robot_motion_env import RobotMotionEnv
from RL_brain import DeepQNetwork
from RL_brain_DDPG import DDPG
import numpy as np

#np.random.seed(10)
np.random.seed(8)

if __name__ == "__main__":



    #创建环境
    env = RobotMotionEnv("UAV-0", False)
    #创建控制器
    # control_dqn = DeepQNetwork(input_size=env.state_dim, output_size=env.actions_num, hidden_layer_sizes=[20,20],
    #                             dim_action=env.action_dim, learning_rate=0.0001, reward_decay=0.9,e_greedy=0.95,
    #                             e_greedy_increment=0.00002,replace_target_iter=200, memory_size=100000, output_graph=False)
    #
    # ctrl = control_dqn

    control_ddpg = DDPG(action_dim= env.action_dim, state_dim= env.state_dim, critic_hidden_layer_sizes=[20,20],
                            actor_hidden_layer_sizes=[20,20], critic_learning_rate=0.001, actor_learning_rate=0.0001,
                            reward_decay=0.9,memory_capacity=100000,batch_size=256,critic_target_replace="soft",critic_soft_tau=0.2,
                            critic_hard_iter=200,actor_target_replace="soft",actor_soft_tau=0.1,actor_hard_iter=200,actor_explore_var=0.5,
                            actor_explore_decoy_rate=0.9999,output_graph=False,discrete_acton=False)
    ctrl = control_ddpg

    #学习或测试
    env.after_mainloop(display=False, mode="train", episode=5000, predict=1000, ctrl=ctrl)



    #结束
    print("quit program!")




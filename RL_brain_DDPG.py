# -*- coding: utf-8 -*-
"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
wankaifang
"""

import tensorflow.compat.v1 as tf
import numpy as np
import time
import numpy as np
import os
import tf_slim as layers



tf.compat.v1.disable_eager_execution()



###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 critic_hidden_layer_sizes=[30],
                 actor_hidden_layer_sizes=[30],
                 critic_learning_rate=0.001,
                 actor_learning_rate=0.001,
                 reward_decay=0.9,
                 memory_capacity=10000,
                 batch_size=32,
                 critic_target_replace="hard",
                 critic_soft_tau=0.01,
                 critic_hard_iter=500,
                 actor_target_replace="hard",
                 actor_soft_tau=0.01,
                 actor_hard_iter=600,
                 actor_explore_var=3,
                 actor_explore_decoy_rate=0.995,
                 output_graph=False,
                 discrete_acton=False
                 ):
        # 初始化变量
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.discrete_acton = discrete_acton
        self.c_hidden_layer_sizes = critic_hidden_layer_sizes
        self.a_hidden_layer_sizes = actor_hidden_layer_sizes
        self.c_lr = critic_learning_rate
        self.a_lr = actor_learning_rate
        self.gamma = reward_decay
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.c_target_replace = critic_target_replace
        self.c_soft_tau = critic_soft_tau
        self.c_hard_iter = critic_hard_iter
        self.a_target_replace = actor_target_replace
        self.a_soft_tau = actor_soft_tau
        self.a_hard_iter = actor_hard_iter
        self.a_explore_var = actor_explore_var
        self.a_explore_decoy_rate = actor_explore_decoy_rate
        self.a_replace_counter = 0
        self.c_replace_counter = 0
        self.learn_count = 0

        #加epsilon
        self.epsilon = 0.0
        self.epsilon_increment = 0.00001

        self.output_graph = output_graph

        # 创建经验池
        self.memory = np.zeros((self.memory_capacity, self.s_dim * 2 + self.a_dim + 1), dtype=np.float32)
        self.pointer = 0

        # sess对象
        self.sess = tf.Session()

        # 全局变量
        self.S = tf.placeholder(tf.float32, [None, self.s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, self.s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        # 创建actor网络
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, self.a_hidden_layer_sizes, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, self.a_hidden_layer_sizes, scope='target', trainable=False)

        # 创建critic网络
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error, otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, self.c_hidden_layer_sizes, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, self.c_hidden_layer_sizes, scope='target', trainable=False)

        # 网络参数
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # 目标网络软更新函数
        # self.soft_replace = [tf.assign(t, (1 - self.a_soft_tau) * t + self.a_soft_tau * e) for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        # 目标网络软软硬更新函数
        if self.c_target_replace == 'hard':
            self.c_replace_counter = 0
            self.c_hard_replace = [tf.assign(t, e) for t, e in zip(self.ct_params, self.ce_params)]
        else:
            self.c_soft_replace = [tf.assign(t, (1 - self.c_soft_tau) * t + self.c_soft_tau * e) for t, e in
                                   zip(self.ct_params, self.ce_params)]

        if self.a_target_replace == 'hard':
            self.a_replace_counter = 0
            self.a_hard_replace = [tf.assign(t, e) for t, e in zip(self.at_params, self.ae_params)]
        else:
            self.a_soft_replace = [tf.assign(t, (1 - self.a_soft_tau) * t + self.a_soft_tau * e) for t, e in
                                   zip(self.at_params, self.ae_params)]

        # critic训练函数(考虑正则化×××)
        q_target = self.R + self.gamma * q_

        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)

        self.ctrain = tf.train.AdamOptimizer(self.c_lr).minimize(td_error, var_list=self.ce_params)

        # actor训练函数(考虑正则化×××)
        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.a_lr).minimize(a_loss, var_list=self.ae_params)

        # 全局参数初始化
        self.sess.run(tf.global_variables_initializer())

        # tensorbord控制
        if self.output_graph:
            tf.summary.FileWriter('logs/', self.sess.graph)

        # 保存参数
        self.saver = tf.train.Saver()
        self._path = './discrete' if self.discrete_acton else './continuous'

    # 选择行动
    def choose_action(self, s, test=False):
        # 分为测试和学习
        if test:
            a = self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
        else:
            # if np.random.uniform() < self.a_explore_var:
            #    a  = np.random.uniform(-1,1, self.a_dim)
            # else:
            #    a = self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
            if np.random.uniform() > self.epsilon:
                a = np.random.uniform(-1,1, self.a_dim)
            else:
                a = self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
                a = np.clip(np.random.normal(a, self.a_explore_var), -1.0,1.0)  # add randomness to action selection for exploration

            # print(a)

        return a

    # 学习过程
    def learn(self):

        # 软更新目标网络
        # self.sess.run(self.soft_replace)

        # 软硬更新目标网络
        if self.c_target_replace == 'soft':
            self.sess.run(self.c_soft_replace)
        else:
            if self.c_replace_counter % self.c_hard_iter == 0:
                self.sess.run(self.c_hard_replace)
            self.c_replace_counter += 1

        if self.a_target_replace == 'soft':
            self.sess.run(self.a_soft_replace)
        else:
            if self.a_replace_counter % self.a_hard_iter == 0:
                self.sess.run(self.a_hard_replace)
            self.a_replace_counter += 1

        # 从经验池采样用于训练
        indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

        # 更新探测率
        self.a_explore_var = self.a_explore_var * self.a_explore_decoy_rate if self.a_explore_var > 0.001 else 0.001
        # epsilon
        if self.epsilon < 0.9:
            self.epsilon += self.epsilon_increment
        else :
            self.epsilon = 0.9
        #print('epsilon',self.epsilon)
        self.learn_count += 1
        #print(self.learn_count)

    # 储存样本
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    # 创建actor网络
    def _build_a(self, s, hidden_layer_sizes, scope, trainable):
        regularizer = layers.l2_regularizer(0.005)
        with tf.variable_scope(scope):
            # 初始化设定
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)

            # 输入层
            prev_layer = s
            # 遍历隐藏层
            for i, layer_size in enumerate(hidden_layer_sizes):
                i += 1
                with tf.variable_scope(('l' + str(i))):
                    li = tf.layers.dense(prev_layer, layer_size, activation=tf.nn.relu, kernel_initializer=init_w,
                                         bias_initializer=init_b,
                                         name=('l' + str(i)), trainable=trainable)
                    prev_layer = li

            # 输出层
            with tf.variable_scope('a'):
                a = tf.layers.dense(prev_layer, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                    bias_initializer=init_b, name='a', trainable=trainable)
        return a

    # 创建critic网络
    def _build_c(self, s, a, hidden_layer_sizes, scope, trainable):
        regularizer = layers.l2_regularizer(0.001)
        with tf.variable_scope(scope):  # eval or target

            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            # 遍历隐藏层
            prev_layer = tf.placeholder(tf.float32, [None, self.s_dim + self.a_dim], name='s_a')
            prev_size = self.s_dim + self.a_dim
            for i, layer_size in enumerate(hidden_layer_sizes):
                i += 1
                with tf.variable_scope(('l' + str(i))):
                    if i == 1:  # 输入层
                        wsi = tf.get_variable(('ws' + str(i)), [self.s_dim, layer_size], initializer=init_w,
                                              trainable=trainable)
                        wai = tf.get_variable(('wa' + str(i)), [self.a_dim, layer_size], initializer=init_w,
                                              trainable=trainable)
                        tf.add_to_collection(tf.GraphKeys.WEIGHTS, wsi)
                        tf.add_to_collection(tf.GraphKeys.WEIGHTS, wai)
                        bi = tf.get_variable(('b' + str(i)), [1, layer_size], initializer=init_b, trainable=trainable)
                        li = tf.nn.relu(tf.matmul(s, wsi) + tf.matmul(a, wai) + bi)
                        prev_layer = li
                        prev_size = layer_size
                    else:
                        wi = tf.get_variable(('w' + str(i)), [prev_size, layer_size], initializer=init_w,
                                             trainable=trainable)
                        tf.add_to_collection(tf.GraphKeys.WEIGHTS, wi)
                        bi = tf.get_variable(('b' + str(i)), [1, layer_size], initializer=init_b, trainable=trainable)
                        li = tf.nn.relu(tf.matmul(prev_layer, wi) + bi)
                        prev_layer = li
                        prev_size = layer_size

            # 输出层
            with tf.variable_scope('q'):
                q = tf.layers.dense(prev_layer, 1, kernel_initializer=init_w, bias_initializer=init_b,
                                    trainable=trainable)  # Q(s,a)

        return q

    def save(self,name):
        if os.path.isdir(self._path) is not True:
            os.mkdir(self._path)
        ckpt_path = os.path.join(self._path, f'DDPG_{name}.ckpt')
        save_path = self.saver.save(self.sess, ckpt_path, write_meta_graph=False)
        print("model saved to file: {}".format(save_path))

    def load(self,name):
        ckpt_path = os.path.join(self._path, f'DDPG_{name}.ckpt')
        self.saver.restore(self.sess, ckpt_path)
        print("model restored from: {}".format(ckpt_path))



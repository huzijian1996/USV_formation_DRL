# -*- coding: utf-8 -*-
#RL-DQN：经典DQN,适用于连续状态-离散行动问题

import numpy as np
import tensorflow.compat.v1 as tf
import os
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()

np.random.seed(1)
tf.set_random_seed(1)

class DeepQNetwork:

    def __init__(self, input_size, output_size, hidden_layer_sizes, dim_action, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, replace_target_iter=300,
                 memory_size=500, batch_size=32, e_greedy_increment=None, output_graph=False, discrete_acton=True):
        self.n_actions = output_size                    # 行动空间大小(输出节点数)
        self.n_features = input_size                    # 状态维数(输入节点数)
        self.hidden_layer_sizes = hidden_layer_sizes    # 隐藏层大小
        self.dim_action = dim_action                    # 行动维数
        self.lr = learning_rate                         # 学习率
        self.gamma = reward_decay                       # 折扣因子
        self.epsilon_max = e_greedy                     # 探索利用率
        self.replace_target_iter = replace_target_iter  # 300次更新一次目标网络
        self.memory_size = memory_size                  # 经验空间大小
        self.batch_size = batch_size                    # 每次训练样本数
        self.epsilon_increment = e_greedy_increment     # 可变探索利用率
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max  #可控制e衰减
        self.learn_step_counter = 0       #学习计数器
        self._discrete_action = discrete_acton

        # 初始化经验池 [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + self.dim_action + 1))

        # 构建两个网络：Q目标网络和Q估计网络
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # 设置sess和saver
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        #日志文件
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        self._path = './discrete' if discrete_acton else './continuous'

    def _build_net(self):
        # ——————————————创建Q估计网络——————————————————————
        with tf.variable_scope('eval_net'):
            # 名称和初始器
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_initializer = tf.random_normal_initializer(0., 0.3)
            b_initializer = tf.constant_initializer(0.1)

            # 输入层
            self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # 输入（状态值）
            self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # 输出（标签值）

            # 遍历创建隐藏层
            prev_layer = self.s
            prev_size = self.n_features
            for i, layer_size in enumerate(self.hidden_layer_sizes):
                i += 1
                with tf.variable_scope(('l'+str(i))):
                    w_i = tf.get_variable(('w'+str(i)), [prev_size, layer_size], initializer=w_initializer, collections=c_names)
                    b_i = tf.get_variable(('b'+str(i)), [1, layer_size], initializer=b_initializer, collections=c_names)
                    l_i = tf.nn.relu(tf.matmul(prev_layer, w_i) + b_i)

                    prev_layer = l_i
                    prev_size = layer_size

            # 输出层
            i = len(self.hidden_layer_sizes)+1
            with tf.variable_scope(('l'+str(i))):
                w_ii = tf.get_variable(('w'+str(i)), [prev_size, self.n_actions], initializer=w_initializer, collections=c_names)
                b_ii = tf.get_variable(('b'+str(i)), [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(prev_layer, w_ii) + b_ii

            # 设置训练器
            with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ————————————————————创建Q目标网络————————————————————————
        with tf.variable_scope('target_net'):
            # 名称和初始器
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # 输入层
            self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')

            # 遍历创建隐藏层
            prev_layer = self.s_
            prev_size = self.n_features
            for i, layer_size in enumerate(self.hidden_layer_sizes):
                i += 1
                with tf.variable_scope(('l' + str(i))):
                    w_i = tf.get_variable(('w' + str(i)), [prev_size, layer_size], initializer=w_initializer,collections=c_names)
                    b_i = tf.get_variable(('b' + str(i)), [1, layer_size], initializer=b_initializer,collections=c_names)
                    l_i = tf.nn.relu(tf.matmul(prev_layer, w_i) + b_i)

                    prev_layer = l_i
                    prev_size = layer_size

            # 输出层
            i = len(self.hidden_layer_sizes) + 1
            with tf.variable_scope(('l' + str(i))):
                w_ii = tf.get_variable(('w' + str(i)), [prev_size, self.n_actions], initializer=w_initializer,collections=c_names)
                b_ii = tf.get_variable(('b' + str(i)), [1, self.n_actions], initializer=b_initializer,collections=c_names)
                self.q_next = tf.matmul(prev_layer, w_ii) + b_ii

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # 存储一个样本
        transition = np.hstack((s, [a, r], s_))

        # 替换经验池中的一条样本
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation, test=False):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if (np.random.uniform() < self.epsilon) or test:     #测试阶段强行使用模型选择行动wkf
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # 每学习一段时间后就将估值网络参数赋给目标网络Fixed-Target-Network
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # 从经验池采样一批样本用于本次训练
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 计算q-next(目标网络的输出) 和 q-eval（估值网络的输出）
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = (self.epsilon + self.epsilon_increment) if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        #print(self.epsilon)

    def save(self,name):
        if os.path.isdir(self._path) is not True:
            os.mkdir(self._path)
        ckpt_path = os.path.join(self._path, f'DQN_{name}.ckpt')
        save_path = self.saver.save(self.sess, ckpt_path, write_meta_graph=False)
        print("model saved to file: {}".format(save_path))

    def load(self,name):
        ckpt_path = os.path.join(self._path, f'DQN_{name}.ckpt')
        self.saver.restore(self.sess, ckpt_path)
        print("model restored from: {}".format(ckpt_path))

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()




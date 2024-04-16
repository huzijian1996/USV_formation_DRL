import matplotlib.pyplot as plt
import numpy as np
from collections import deque


def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)


def average(list_a,size=100):
    ave_deque = deque(maxlen=size)
    ave_list = []
    for i in range(len(list_a)):
        ave_deque.append(list_a[i])
        ave_list.append(averagenum(ave_deque))
    return ave_list

if __name__ == "__main__":



    a = np.load('dqn1.npy')
    hit_rate_dqn = a.tolist()
    ave_hit_rate_dqn = average(hit_rate_dqn,500)

    b = np.load('dqn_rbo3.npy')
    hit_rate_dqn_rbo = b.tolist()
    ave_hit_rate_dqn_rbo = average(hit_rate_dqn_rbo, 500)

    fig1 = plt.figure("hit rate")
    plt.plot(np.arange(len(ave_hit_rate_dqn)), ave_hit_rate_dqn, label='DQN',linewidth=4,color='blue')

    plt.plot(np.arange(len(ave_hit_rate_dqn_rbo)), ave_hit_rate_dqn_rbo, label='DQN with MSR',linewidth=4,color='red')

    plt.legend(loc='upper left')
    plt.ylabel('Hit Rate')
    plt.xlabel('Episode')
    fig1.savefig('dqn.png')
    fig1.show()


    r1 = np.load('reward_dqn1.npy')
    r1 = r1
    reward_dueling_dqn = r1.tolist()
    ave_reward_dueling_dqn = average(reward_dueling_dqn,500)

    r2 = np.load('reward_dqn_rbo3.npy')
    r2 = r2
    reward_dueling_dqn_rbo = r2.tolist()
    ave_reward_dueling_dqn_rbo = average(reward_dueling_dqn_rbo,500)


    fig2 = plt.figure("average rewards")
    plt.plot(np.arange(len(reward_dueling_dqn_rbo)), reward_dueling_dqn_rbo, linewidth=0.4, color='lightcoral')
    plt.plot(np.arange(len(reward_dueling_dqn)), reward_dueling_dqn, linewidth=0.2, color='dodgerblue')


    plt.plot(np.arange(len(ave_reward_dueling_dqn)), ave_reward_dueling_dqn,label='Dueling DQN', linewidth=3, color='blue')
    plt.plot(np.arange(len(ave_reward_dueling_dqn_rbo)), ave_reward_dueling_dqn_rbo,label='Dueling DQN with MSR', linewidth=3, color='red')
    plt.legend(loc='lower right')
    plt.ylim(-0.6, 0.6)
    plt.ylabel('Episode Average Reward')
    plt.xlabel('Episode')
    #fig2.savefig('duelingdqn_r.png')
    fig2.show()


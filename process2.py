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
    aa = average(hit_rate_dqn,500)

    b = np.load('dqn_rbo3.npy')
    hit_rate_dqn_rbo = b.tolist()
    bb = average(hit_rate_dqn_rbo, 500)

    for i in range(len(aa)):
        if aa[i] >= 0.8:
            print('dqn',i)
            break
    for i in range(len(bb)):
        if bb[i] >= 0.8:
            print('dqn_msr',i)
            break


    r1 = np.load('reward_dqn1.npy')
    r1 = r1
    reward_dueling_dqn = r1.tolist()
    cc = average(reward_dueling_dqn,500)

    r2 = np.load('reward_dqn_rbo3.npy')
    r2 = r2
    reward_dueling_dqn_rbo = r2.tolist()
    dd = average(reward_dueling_dqn_rbo,500)
    count = 0
    sum = 0
    for i in range(1000):
        if 4000+i <= len(cc):
            count += 1
            sum += cc[4000+i]
    print('dqn',sum/count)
    count = 0
    sum = 0
    for i in range(1000):
        if 4000+i <= len(dd):
            count += 1
            sum += dd[4000+i]
    print('dqn',sum/count)
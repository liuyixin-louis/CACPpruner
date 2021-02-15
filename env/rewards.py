# Code for "Conditional Automated Channel Pruning for Deep Neural Networks"
# implement by Yixin Liu
# seyixinliu@mail.scut.edu.cn

import numpy as np


# for pruning
def acc_reward(net, acc, flops):
    return acc * 0.01


def acc_flops_reward(net, acc, flops):
    error = (100 - acc) * 0.01
    return -error * np.log(flops)

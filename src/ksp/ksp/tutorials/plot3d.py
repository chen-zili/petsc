#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import sys
import os
import re
import math

import matplotlib.font_manager as fm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d


if __name__ == '__main__':

    if 2 != len(sys.argv):
        print("Error: 参数数量错误！")
        print("argv: ", len(sys.argv))
        exit(-1)

    if not os.path.exists(sys.argv[1]):
        print("文件不存在: ", sys.argv[1])

    with open('./' + sys.argv[1], 'r') as f:
        print("打开: ", sys.argv[1])
        VecList = []
        while (True):
            line = f.readline()
            if not line:
                break;
            else:
                numlist = re.findall('\d*[.]\d*', line)
                if 0 != len(numlist):
                    VecList.append(float(numlist[0]))

    size_2d = int(math.sqrt(len(VecList)))
    VecList = np.array(VecList)
    poissonResult = np.reshape(VecList, (size_2d, size_2d))

    x = np.linspace(-1, 1, 65)
    y = np.linspace(-1, 1, 65)
    xx, yy = np.meshgrid(x, y)
    zz = -0.25*xx**2 + 0.25 - 0.25*yy**2 + 0.25

    fig = plt.figure(0)
    ax3d_1 = Axes3D(fig)
    ax3d_1.plot_surface(xx, yy, poissonResult)
    ax3d_1.set_zlim3d((0, 1))
    ax3d_1.set_xlim3d((-1, 1))
    ax3d_1.set_ylim3d((-1, 1))
    plt.title("Petsc result")    

    fig = plt.figure(1)
    ax3d_2 = Axes3D(fig)
    ax3d_2.plot_surface(xx, yy, zz)
    ax3d_2.set_zlim3d((0, 1))
    ax3d_2.set_xlim3d((-1, 1))
    ax3d_2.set_ylim3d((-1, 1))
    plt.title("Parse result")

    plt.show()
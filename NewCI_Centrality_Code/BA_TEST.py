__author__='SONG Jiaxiu'
# -*- coding: utf-8 -*-
"""
This module implements NewCI.
"""

import networkx as nx                   #导入networkx包
import matplotlib.pyplot as plt     #导入绘图包matplotlib（需要安装，方法见第一篇笔记）
G =nx.random_graphs.barabasi_albert_graph(20,1)   #生成一个BA无标度网络G
nx.draw(G)                          #绘制网络G
#plt.savefig("ba.png")           #输出方式1: 将图像存为一个png格式的图片文件
plt.show()  
raw_input() 
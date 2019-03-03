__author__='SONG Jiaxiu'
# -*- coding: utf-8 -*-
"""
This module implements NewCI.
"""

import linecache
import string
import os
import math
import time
import networkx as nx
import sys
from collections import Counter
import operator
import networkx as nx             
import matplotlib.pyplot as plt
from networkx.generators.atlas import *
import numpy as np
import random
import requests
import pandas as pd
import csv
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic as isomorphic
import linecache
import matplotlib
from Celegans import max_connected_component,connected_components_nums
matplotlib.rcParams['font.family'] = 'sans-serif'  
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'# 中文设置成宋体，除此之外的字体设置成New Roman

##########********************数据存放文件****************

#*******************************************************************************************#
lenght=len(max_connected_component['ci'])
#截取
remove_rate=int(lenght)
remove_rate1=int(lenght)
# fig=plt.figure()
# ax=fig.add_subplot(1,1,1)
# #ax.plot(np.array(remove_rate,dtype=float) , np.array(connected_components_nums['degree']),'k--',label='DC')
# ax.plot(np.arange(lenght,dtype=float)/np.array(lenght,dtype=float),np.array(connected_components_nums['between']),'c--',label='BC')
# ax.plot(np.arange(lenght,dtype=float)/np.array(lenght,dtype=float) ,np.array(connected_components_nums['closeness']),'y--',label='CC')
# ax.plot(np.arange(lenght,dtype=float)/np.array(lenght,dtype=float) ,np.array(connected_components_nums['pagerank']),'m--',label='EC')
# ax.plot(np.arange(lenght,dtype=float)/np.array(lenght,dtype=float) ,np.array(connected_components_nums['kcore']),'b--',label='k-core')
# ax.plot(np.arange(lenght,dtype=float)/np.array(lenght,dtype=float) ,np.array(connected_components_nums['ci']),'g-',label='CI')
# ax.plot(np.arange(lenght,dtype=float)/np.array(lenght,dtype=float) ,np.array(connected_components_nums['newci']),'r-',label='NewCI')
# #ax.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['newci91']),'y-',label='NewCI_B/R')
# #ax.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['newci19']),'k-',label='NewCI_S/R')
# #ax.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['newci_s']),'c-',label='NewCI_B')
# #ax.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['newci_b']),'m-',label='NewCI_S')




# ax.legend(loc='best')
# ax.set_xlabel("q",fontstyle='italic')
# ax.set_ylabel("C(q)",fontstyle='italic')
# fig.show()


fig3=plt.figure()
ax3=fig3.add_subplot(2,1,2)
#ax.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['degree']),'k--',label='DC')
#ax3.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['between']),'c--',label='BC')
#ax3.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['closeness']),'y--',label='CC')
#ax3.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['pagerank']),'m--',label='EC')
#ax3.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['kcore']),'b--',label='k-core')
ax3.plot(np.arange(lenght,dtype=float)/np.array(lenght,dtype=float),np.array(connected_components_nums['ci']),'g-',label='CI')
#ax3.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['newci']),'r-',label='NewCI')
ax3.plot(np.arange(lenght,dtype=float)/np.array(lenght,dtype=float) ,np.array(connected_components_nums['newci91']),'y-',label='NewCI_B/R')
ax3.plot(np.arange(lenght,dtype=float)/np.array(lenght,dtype=float) ,np.array(connected_components_nums['newci19']),'k-',label='NewCI_S/R')
ax3.plot(np.arange(lenght,dtype=float)/np.array(lenght,dtype=float) ,np.array(connected_components_nums['newci_s']),'c-',label='NewCI_B')
ax3.plot(np.arange(lenght,dtype=float)/np.array(lenght,dtype=float) ,np.array(connected_components_nums['newci_b']),'m-',label='NewCI_S')

#ax3.legend(loc='best')
ax3.set_xlabel("q",fontstyle='italic')
ax3.set_ylabel("C(q)",fontstyle='italic')
#fig3.show()

arr=np.arange(lenght,dtype=float)/np.array(lenght,dtype=float)

#fig1=plt.figure()
ax1=fig3.add_subplot(2,1,1)
#ax1.plot(np.array(remove_rate,dtype=float) ,np.array(max_connected_component['degree'],dtype=float)/float(G_init.number_of_nodes()),'k--',label='DC')
#ax1.plot(np.array(remove_rate,dtype=float) ,np.array(max_connected_component['between'],dtype=float)/float(G_init.number_of_nodes()),'c--',label='BC')
#ax1.plot(np.array(remove_rate,dtype=float) ,np.array(max_connected_component['closeness'],dtype=float)/float(G_init.number_of_nodes()),'y--',label='CC')
#ax1.plot(np.array(remove_rate,dtype=float) ,np.array(max_connected_component['pagerank'],dtype=float)/float(G_init.number_of_nodes()),'m--',label='EC')
#ax1.plot(np.array(remove_rate,dtype=float) ,np.array(max_connected_component['kcore'],dtype=float)/float(G_init.number_of_nodes()),'b--',label='k-core')
ax1.plot(arr[0:remove_rate1] ,np.array(max_connected_component['ci'],dtype=float)[0:remove_rate1]/float(lenght),'g-',label='CI')
#ax1.plot(np.array(remove_rate,dtype=float) ,np.array(max_connected_component['newci'],dtype=float)/float(G_init.number_of_nodes()),'r-',label='NewCI')
ax1.plot(arr[0:remove_rate1] ,np.array(max_connected_component['newci91'],dtype=float)[0:remove_rate1]/float(lenght),'y-',label='NewCI_B/R')
ax1.plot(arr[0:remove_rate1],np.array(max_connected_component['newci19'],dtype=float)[0:remove_rate1]/float(lenght),'k-',label='NewCI_S/R')
ax1.plot(arr[0:remove_rate1] ,np.array(max_connected_component['newci_s'],dtype=float)[0:remove_rate1]/float(lenght),'c-',label='NewCI_B')
ax1.plot(arr[0:remove_rate1] ,np.array(max_connected_component['newci_b'],dtype=float)[0:remove_rate1]/float(lenght),'m-',label='NewCI_S')

ax1.legend(loc='best')
#l1 = plt.legend([p2, p1], ["line 2", "line 1"], loc='upper left')
ax1.set_xlabel("q",fontstyle='italic')
ax1.set_ylabel("G(q)",fontstyle='italic')
#fig1.show()
fig3.show()

fig2=plt.figure()
ax2=fig2.add_subplot(1,1,1)
#ax1.plot(np.array(remove_rate,dtype=float) ,np.array(max_connected_component['degree'],dtype=float)/float(G_init.number_of_nodes()),'k--',label='DC')
ax2.plot(arr[0:remove_rate] ,np.array(max_connected_component['between'],dtype=float)[0:remove_rate]/float(lenght),'c--',label='BC')
ax2.plot(arr[0:remove_rate],np.array(max_connected_component['closeness'],dtype=float)[0:remove_rate]/float(lenght),'y--',label='CC')
ax2.plot(arr[0:remove_rate] ,np.array(max_connected_component['pagerank'],dtype=float)[0:remove_rate]/float(lenght),'m--',label='EC')
ax2.plot(arr[0:remove_rate],np.array(max_connected_component['kcore'],dtype=float)[0:remove_rate]/float(lenght),'b--',label='k-core')
ax2.plot(arr[0:remove_rate] ,np.array(max_connected_component['ci'],dtype=float)[0:remove_rate]/float(lenght),'g-',label='CI')
ax2.plot(arr[0:remove_rate] ,np.array(max_connected_component['newci'],dtype=float)[0:remove_rate]/float(lenght),'r-',label='NewCI')
##ax2.plot(np.array(remove_rate,dtype=float) ,np.array(max_connected_component['newci91'],dtype=float)/float(G_init.number_of_nodes()),'y-',label='NewCI_B/R')
##ax2.plot(np.array(remove_rate,dtype=float) ,np.array(max_connected_component['newci19'],dtype=float)/float(G_init.number_of_nodes()),'k-',label='NewCI_S/R')
#ax1.plot(np.array(remove_rate,dtype=float) ,np.array(max_connected_component['newci_s'],dtype=float)/float(G_init.number_of_nodes()),'c-',label='NewCI_B')
#ax1.plot(np.array(remove_rate,dtype=float) ,np.array(max_connected_component['newci_b'],dtype=float)/float(G_init.number_of_nodes()),'m-',label='NewCI_S')
left, bottom, width, height = 0.55, 0.55, 0.3, 0.3
# 获得绘制的句柄
ax22= fig2.add_axes([left, bottom, width, height])
ax22.plot(np.arange(lenght,dtype=float)/np.array(lenght,dtype=float) ,np.array(connected_components_nums['ci']),'g-',label='CI')
ax22.plot(np.arange(lenght,dtype=float)/np.array(lenght,dtype=float) ,np.array(connected_components_nums['newci']),'r-',label='NewCI')
ax22.set_xlabel("q",fontstyle='italic')
ax22.set_ylabel("C(q)",fontstyle='italic')
ax2.legend(loc='lower left')
ax2.set_xlabel("q",fontstyle='italic')
ax2.set_ylabel("G(q)",fontstyle='italic')
fig2.show()



raw_input()
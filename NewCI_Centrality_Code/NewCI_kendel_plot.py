__author__='SONG Jiaxiu'
# -*- coding: utf-8 -*-
"""
This module implements result_sort_process.
"""

import csv
import os
import numpy as np
import random
import requests
import pandas as pd
import time
import networkx as nx
import sys
from collections import Counter
import operator
from kendall_tau import cal_kendall_tau
# file_node = open(r"C:\Users\Administrator\Desktop\3th_result\SIR_matlab\test_node.txt", "w+")
# file_value = open(r"C:\Users\Administrator\Desktop\3th_result\SIR_matlab\test_value.txt", "w+")
# file_dict = open(r"C:\Users\Administrator\Desktop\3th_result\SIR_matlab\test_dict.txt", "w+")
import linecache
import string
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.ticker import FuncFormatter 
matplotlib.rcParams['font.family'] = 'sans-serif'  
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'# 中文设置成宋体，除此之外的字体设置成New Roman
matplotlib.rcParams['font.family'] = 'sans-serif'  
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'# 中文设置成宋体，除此之外的字体设置成New Roman
sns.set_style('whitegrid') 

bc=[0.682132483139098, 0.6713874734007031, 0.6659672393350852, 0.6643781233945169, 0.6544506963546365, 0.6512329700998093, 0.6492752946304858, 0.6424153185729257, 0.6461015693687537, 0.6410434216876553]
cc=[0.6079169012157221, 0.6253602222828567, 0.6513344100056784, 0.6676046170725491, 0.6803618733443295, 0.6935175967100039, 0.7085674112421309, 0.718444036390542, 0.7278128884398657, 0.7348680420869089]
ec=[0.5163570619135761, 0.5209301334696791, 0.528874566033036, 0.5274095050338938, 0.5257134593056365, 0.5301681296792025, 0.5315109383846801, 0.5277926496218127, 0.5288442487752237, 0.5301728821142109]
k_core=[0.6537249667534396, 0.6520266267462126, 0.6398985764818379, 0.6422007215507359, 0.6323921053860657, 0.6257124760432209, 0.6205709968724061, 0.6112181228373348, 0.608719653039469, 0.6031415234176235]
ci=[0.7690789371261042, 0.7765991738956939, 0.771324134913436, 0.7767007776786321, 0.762468955537693, 0.7563506051570474, 0.7487486756684341, 0.7352633135779526, 0.7328431770191908, 0.7207581444854957]
newci=[0.7698499787369503, 0.7774737858143093, 0.7705966846030119, 0.7764741356918521, 0.7631126646657277, 0.7567081849221625, 0.7488116044630281, 0.7357046345254571, 0.7329846029299582, 0.7209408674177152]
labels = ['0.1','0.15','0.2','0.25','0.3','0.35','0.4','0.45','0.5','0.55']
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
#ax.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['degree']),'k--',label='DC')
ax.plot(np.arange(10) ,np.array(bc),'c--',label='BC')
ax.plot(np.arange(10) ,np.array(cc),'y--',label='CC')
ax.plot(np.arange(10) ,np.array(ec),'m--',label='EC')
ax.plot(np.arange(10) ,np.array(k_core),'b--',label='k-core')
ax.plot(np.arange(10) ,np.array(ci),'g-',label='CI')
ax.plot(np.arange(10) ,np.array(newci),'r-',label='NewCI+')
plt.tick_params(labelsize=7.5)
x=np.arange(10)
y=np.array(labels)
plt.xticks(x,labels)
#ax.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['newci91']),'y-',label='NewCI_B/R')
#ax.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['newci19']),'k-',label='NewCI_S/R')
#ax.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['newci_s']),'c-',label='NewCI_B')
#ax.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['newci_b']),'m-',label='NewCI_S')
ax.legend(loc='lower right',bbox_to_anchor=(0.9,0.08),ncol=3,fontsize=7.5,frameon=False)
# ax.set_xlabel("rate",fontstyle='italic')
# ax.set_ylabel("kendall_value",fontstyle='italic')
ax.set_xlabel("rate",fontsize=7.5)
ax.set_ylabel("kendall_value",fontsize=7.5)
fig.subplots_adjust(left=0.17, bottom=0.19, right=0.95, top=0.95, hspace=0.2, wspace=0.25)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(3,2.1)
fig.savefig(r'C:\Python27\sjxwork\NewCI_Centrality_Code\Nest1png.png', dpi=600)
fig.show()
raw_input()

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
file_node = open(r"C:\Users\Administrator\Desktop\3th_result\SIR_matlab\test_node.txt", "w+")
file_value = open(r"C:\Users\Administrator\Desktop\3th_result\SIR_matlab\test_value.txt", "w+")
file_dict = open(r"C:\Users\Administrator\Desktop\3th_result\SIR_matlab\test_dict.txt", "w+")
csv_data = pd.read_csv(r'C:\Users\Administrator\Desktop\3th_result\SIR_matlab\citeseer\citeseer1.8.csv')
#print csv_data
#indexs_list=[0.1,0.15,0.20,0.25,0.3,0.35,0.4,0.45,0.5,0.55]
# for id in range(10):

#     index=indexs_list[id]
#     print index
for id in range(1,11):
    index=round(float(id)/100.0,2)
    print index
    p001= list(csv_data[str(index)])
    print p001
    dict_p001={}
    for i in range (len(p001)):
        #print i+1
        dict_p001[i+1]=p001[i]
    #print dict_p001
    file_dict.write(str(dict_p001)+'\n')
    dict_p001_sort=sorted(dict_p001.items(),key=operator.itemgetter(1),reverse=True)
    print dict_p001_sort
    dict_p001_node= [x for x,_ in dict_p001_sort]
    dict_p001_value=[x for _,x in dict_p001_sort]
    print dict_p001_node
    file_node.write(str(dict_p001_node)+'\n')
    file_value.write(str(dict_p001_value)+'\n')



'''
newCI_sort=sorted(newci.items(),key=operator.itemgetter(1),reverse=True)
#print newCI_sort
res_list_NewCI = [x for x,_ in newCI_sort]
print res_list_NewCI
file.write("time_newci"+str(time_newci)+'\n')
file.write(str(res_list_NewCI)+'\n')
'''
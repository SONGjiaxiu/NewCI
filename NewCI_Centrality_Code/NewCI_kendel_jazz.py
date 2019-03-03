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

##########********************数据存放文件****************
#file = open(r"C:\Users\Administrator\Desktop\3th_result\karate.txt", "w+")
file = open(r"C:\Users\Administrator\Desktop\3th_result\Celegans.txt", "w+")
#file = open(r"C:\Python27\sjxwork\NewCI_Centrality_Code\result\email.txt", "w+")
#file = open(r"C:\Users\Administrator\Desktop\3th_result\jazz.txt", "w+")
#file = open(r"C:\Users\Administrator\Desktop\3th_result\power.txt", "w+")
#file = open(r"C:\Users\Administrator\Desktop\3th_result\Router.txt", "w+")
#file = open(r"C:\Users\Administrator\Desktop\3th_result\Yeast.txt", "w+")


#*******************************************************************************************#
##构建网络
def createGraph(filename) :
    G = nx.Graph()
    for line in open(filename) :
        strlist = line.split(',', 3)
        #n1 = int(strlist[0])
        #n2 = int(strlist[1])
        n1 = strlist[0]
        n2 = strlist[1]
        #weight = float(strlist[2])
        #G.add_weighted_edges_from([(n1, n2)]) 
        G.add_edge(n1,n2)
    return G
##构建网络的方式2
def make_link(G, node1, node2):
    if node1 not in G:
        G[node1] = {}
    (G[node1])[node2] = 1
    if node2 not in G:
        G[node2] = {}
    (G[node2])[node1] = 1
    return G
#*******************************************************************************************#
##CI中心性的2阶实现
def Collective_Influence(G, l=2):
    Collective_Influence_Dic = {}
    node_set = G.nodes()
    for nid in node_set:
        CI = 0
        neighbor_set = []
        neighbor_hop_1 = G.neighbors(nid)
        neighbor_hop_2 = []
        for nnid in neighbor_hop_1:
            neighbor_hop_2  = list(set(neighbor_hop_2).union(set(G.neighbors(nnid))))
            #print '2_hop:', nnid, G.neighbors(nnid)
        #end for

        center = [nid]
        neighbor_set = list(   set(neighbor_hop_2).difference(   set(neighbor_hop_1).union(set(center))  )    )
        #print nid, neighbor_hop_1, neighbor_hop_2, neighbor_set

        total_reduced_degree = 0
        for id in neighbor_set:
            total_reduced_degree = total_reduced_degree + (G.degree(id)-1.0)
        #end

        CI = (G.degree(nid)-1.0) * total_reduced_degree
        Collective_Influence_Dic[nid] =round(CI,3) 
    #end for
    #print "Collective_Influence_Dic:",sorted(Collective_Influence_Dic.iteritems(), key=lambda d:d[1], reverse = True)

    return Collective_Influence_Dic



##********************************l阶邻域度均衡性******************************************
def l_Balance(G): #
    l_Balance_Dic = {}
    node_set = G.nodes()
    for nid in node_set:
        degree = G.degree(nid)
        Neighbor_Set = G.neighbors(nid)
        #degree_max = 0
        #degree_min = 0
        #print type(Neighbor_Set)
        seq_neibhood=[]
        for ngb in Neighbor_Set:
            #print ngb
            seq_neibhood.append(G.degree(ngb))
        #print seq_neibhood
        degree_max=max(seq_neibhood)
        #print degree_max
        degree_min=min(seq_neibhood)
        #print degree_min
        #total_neighbor_degree = total_neighbor_degree + G.degree(ngb)
        l_Balance_Dic[nid] =round(float(1.0 / float(float(degree_max-degree_min)/(degree_max+degree_min)+1) ),3 ) 
        #print l_Balance_Dic[nid]  
    #end for
    return l_Balance_Dic

##********************************************l阶邻域簇间连接强度************************************************************##
def l_Connection_strength(G):
    l_Connection_strength_Dic={}
    node_set=G.nodes()
    Connection_num=0

    #_l阶连通图的数量
      
    #print nid,i_2_nei   
    for nid in node_set:
       
        degree=G.degree(nid)
        Neighbor_Set=G.neighbors(nid)
        #print nid,Neighbor_Set
        #print len(Neighbor_Set)
   
        
        # i__nei=set(G.neighbors(i))
       
        ###current_1_neighbor=G.neighbors(nid)
        #print nid,current_1_neighbor
        ###current_2_neighbor=[]
        ###for nnid in current_1_neighbor:
            ###current_2_neighbor = list(set(current_2_neighbor).union(set(G.neighbors(nnid))))
        #print '2_hop:', nid,current_2_neighbor
        ###current_2_neighbor= list(  set(current_2_neighbor).difference( set(current_1_neighbor).union(set([nid]))  ) ) 
        #print nid ,current_2_neighbor 
        #print nid,Neighbor_Set
        
        if len(Neighbor_Set)==1:
            Connection_num=1
            #print nid 
            l_Connection_strength_Dic[nid]=1.0
            #print nid,l_Connection_strength_Dic[nid]

        elif len(Neighbor_Set)>1:
            G_conn=nx.Graph()
            #print nid, Neighbor_Set
            ##vi,j组合
            Cluster_head_connection_set=[]
            for i in range(0,len(Neighbor_Set)):
                #vi目标节点的邻居
                vi=Neighbor_Set[i]
                #print nid,Neighbor_Set[i]
                n_vi_2=[]
                ##n_vi 是vi的邻居
                for n_vi in G.neighbors(vi):
                    n_vi_2= list(set(n_vi_2).union(set(G.neighbors(n_vi))))
                n_vi_2=list(set(n_vi_2).difference(set(G.neighbors(vi)).union(set([nid]))))
                for j in range(i+1,len(Neighbor_Set)):
                    vj=Neighbor_Set[j]
                    #print vi,vj
                    fai_ij=list(set(n_vi_2).intersection(set(G.neighbors(vj))))
                    #print vi,vj,fai_ij
                    if fai_ij:
                        Cluster_head_connection_set.append(list([vi,vj]))
                        #
            #print nid,Cluster_head_connection_set
            for k in Cluster_head_connection_set:
                G_conn.add_edge(k[0],k[1])
            H=len(list(nx.connected_components(G_conn)))
            #print nid,H
            G_conn_nodenums=int(nx.number_of_nodes(G_conn))
            ##独立簇的数量
            independent_cluster_num=int(len(Neighbor_Set))-int(G_conn_nodenums )
            ##l-阶的连通数
            Connection_num=int(H)+int(independent_cluster_num)
            l_Connection_strength_Dic[nid]=round(float(Connection_num)/float(len(Neighbor_Set)),3)
            #print nid,l_Connection_strength_Dic[nid]
    return l_Connection_strength_Dic









##********************************************计算网络中节点局部聚类系数************************************************************##
def clustering_coefficient(G,v):
    neighbors = G[v].keys()
    if len(neighbors) == 1: return 0.0
    links = 0
    for w in neighbors:
        for u in neighbors:
            if u in G[w]: links += 0.5
    return round(2.0*links/(len(neighbors)*(len(neighbors)-1)),3)


##*********************************************************************************************


##*************************************CI(l变量时候)******************************************************##
#from __future__ import print_function
def Collective_Influenc(G, l):
    Collective_Influence_Dic = {}
    node_set = G.nodes()
    for nid in node_set:
        CI = 0
        neighbor_set = []
        neighbor_hop_1 = G.neighbors(nid)
        neighbor_hop_2 = []
        for nnid in neighbor_hop_1:
            neighbor_hop_2  = list(set(neighbor_hop_2).union(set(G.neighbors(nnid))))
            #print '2_hop:', nnid, G.neighbors(nnid)
        #end for

        center = [nid]
        neighbor_set = list(   set(neighbor_hop_2).difference(   set(neighbor_hop_1).union(set(center))  )    )
        #print nid, neighbor_hop_1, neighbor_hop_2, neighbor_set

        total_reduced_degree = 0
        for id in neighbor_set:
            total_reduced_degree = total_reduced_degree + (G.degree(id)-1.0)
        #end

        CI = (G.degree(nid)-1.0) * total_reduced_degree
        Collective_Influence_Dic[nid] = CI
    #end for
    #print "Collective_Influence_Dic:",sorted(Collective_Influence_Dic.iteritems(), key=lambda d:d[1], reverse = True)

    return Collective_Influence_Dic


##*****************************************lll********************************************************##

def New_Collective_Influence(G):
    G_CI_value=Collective_Influence(G)
    G_Balance=l_Balance(G)
    G_Strength=l_Connection_strength(G)
    G_c=nx.clustering(G_init)
    ##等价
    # G_Robustness={} 
    # for (x,y) in G_init.edges(): make_link(G_R,x,y)
    # #print G_R
    # for v in G_R.keys():
    #     clustering_coefficient(G_R,v), " "+v
    #     G_Robustness[v]=clustering_coefficient(G_R,v)
    newCI={}
    for nid in G_Balance.keys():
        #newCI[nid]=round(float((0.5*G_Balance[nid]+0.5*G_Strength[nid])*G_CI_value[nid])/float(1+G_Robustness[nid]),3)
        #newCI[nid]=round(float((0.5*G_Balance[nid]+0.5*G_Strength[nid])*G_CI_value[nid])/float(1+G_c[nid]),3)
        #newCI[nid]=round(float(0.1*G_Balance[nid]/float(1+G_c[nid])*G_CI_value[nid]+0.9*G_Strength[nid]*G_CI_value[nid]),3)
        #newCI[nid]=round(float(0.1*G_Balance[nid]/float(1+G_c[nid])*G_CI_value[nid]+0.9*G_Strength[nid]*G_CI_value[nid]),3)
        #print nid, newCI
        #newCI91[nid]=round(float((1.0*G_Balance[nid]+0*G_Strength[nid])*G_CI_value[nid])/float(1+G_c[nid]),3)
        # newCI19[nid]=round(float((0*G_Balance[nid]+1.0*G_Strength[nid])*G_CI_value[nid])/float(1+G_c[nid]),3)
        #newCI[nid]=round(float(1.0*G_Balance[nid]*G_CI_value[nid]),3)
        #newCI[nid]=round(float(1.0*G_Strength[nid]*G_CI_value[nid]),3)
        newCI[nid]=round(float(float(1+G_c[nid])*G_CI_value[nid]),3)
    return newCI
    #G_NewCI=dict(0.5*G_Balance)
    #list_G_CI_value

###*********************************对比方法********************************************
def Degree_Centrality(G):
    Degree_Centrality = nx.degree_centrality(G)
    #print "Degree_Centrality:", sorted(Degree_Centrality.iteritems(), key=lambda d:d[1], reverse = True)
    return Degree_Centrality

def Between_Centrality(G):
    Bet_Centrality = nx.betweenness_centrality(G)
    #print "Bet_Centrality:", sorted(Bet_Centrality.iteritems(), key=lambda d:d[1], reverse = True)
    return Bet_Centrality

def Closeness_Centrality(G):
    Closeness_Centrality = nx.closeness_centrality(G)
    #print "Closeness_Centrality:", sorted(Closeness_Centrality.iteritems(), key=lambda d:d[1], reverse = True)
    return Closeness_Centrality

def Page_Rank(G):	
    PageRank_Centrality = nx.pagerank(G, alpha=0.85)
    #print "PageRank_Centrality:", sorted(PageRank_Centrality.iteritems(), key=lambda d:d[1], reverse = True)
    return PageRank_Centrality

def Eigen_Centrality(G):
    Eigen_Centrality = nx.nx.eigenvector_centrality_numpy(G)
    #print "Eigen_Centrality:", sorted(Eigen_Centrality.iteritems(), key=lambda d:d[1], reverse = True)
    return Eigen_Centrality

def KShell_Centrality(G):
    #网络的kshell中心性
    #The k-core is found by recursively pruning nodes with degrees less than k.
    #The k-shell is the subgraph of nodes in the k-core but not in the (k+1)-core.
    nodes = {}
    core_number = nx.core_number(G) #The core number of a node is the largest value k of a k-core containing that node.
    for k in list(set(core_number.values())):
        nodes[k] = list(n for n in core_number if core_number[n]==k)
    #print core_number #{'1': 2, '0': 2, '3': 2, '2': 2, '4': 1}字典（节点：KShell值）
    #print nodes.keys(),nodes
    KShell_Centrality = core_number
    return KShell_Centrality

###********************************************************************************************************




#G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\karate1.csv")
G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset_csv\Router.csv")
#G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset_csv\Email.csv")
#G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset_csv\Email.csv")
#G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset\Power.csv")
#G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset\Router.csv")
#G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset\Yeast.csv")
#print G_init.edges()

# ###度
# time_degree_start=time.time()
# G_degree=Degree_Centrality(G_init)
# time_degree=time.time()-time_degree_start
# print "degree"
# print time_degree
# Degree_sort=sorted(G_degree.iteritems(), key=lambda d:d[1])
# print Degree_sort
# ##使用解包代替x[0]
# # res_list_d = [x for x,_ in Degree_sort]
# # print res_list_d
# # file.write("degree_time"+str(time_degree)+'\n')
# # file.write(str(res_list_d)+'\n')
# file.write("degree"+'\n')
# id_degree = [x for x,_ in Degree_sort]
# id_degree_value=[x for _,x in Degree_sort]
# # print id_degree
# # print id_degree_value
# file.write(str(id_degree_value)+'\n')
# yingxiangli_value_degree=[]
# print data005
# for i in id_degree:
#     #print type(i)
#     yingxiangli_value_degree.append(data005[int(i)])
# # print yingxiangli_value_degree
# file.write(str(yingxiangli_value_degree)+'\n')

###介数
time_Between_start=time.time()
G_Between=Between_Centrality(G_init)
time_between=time.time()-time_Between_start
print "between"
print time_between
Between_sort=sorted(G_Between.iteritems(), key=lambda d:d[1])
id_between = [x for x,_ in Between_sort]
id_between_value=[x for _,x in Between_sort]
###接近度中心性
time_closeness_start=time.time()
G_Closeness=Closeness_Centrality(G_init)
time_closeness=time.time()-time_closeness_start
print "closeness"
print time_closeness
Closeness_sort=sorted(G_Closeness.iteritems(), key=lambda d:d[1])
id_closeness = [x for x,_ in Closeness_sort]
id_closeness_value=[x for _,x in Closeness_sort]
time_pagerank_start=time.time()
G_Page_Rank=Eigen_Centrality(G_init)
time_pagerank=time.time()-time_pagerank_start
print "pagerank"
print time_pagerank
Page_Rank_sort=sorted(G_Page_Rank.iteritems(), key=lambda d:d[1])
id_pagerank = [x for x,_ in Page_Rank_sort]
id_pagerank_value=[x for _,x in Page_Rank_sort]
time_kshell_start=time.time()
G_KShell=KShell_Centrality(G_init)
time_kshell=time.time()-time_kshell_start
 
KShell_sort=sorted(G_KShell.iteritems(), key=lambda d:d[1])
id_kcore = [x for x,_ in KShell_sort]
id_kcore_value=[x for _,x in KShell_sort]
time_CI_start=time.time()
CI=Collective_Influence(G_init)
time_CI=time.time()-time_CI_start
print "ci"
print time_CI
CI_sort=sorted(CI.items(),key=operator.itemgetter(1))
id_CI = [x for x,_ in CI_sort]
id_CI_value=[x for _,x in CI_sort]
B={}
B=l_Balance(G_init)
G_c=nx.clustering(G_init)
S=l_Connection_strength(G_init)
#print S
time_newci_start=time.time()
newci=New_Collective_Influence(G_init)
time_newci=time.time()-time_newci_start
print "newci"
print time_newci
#print newci

newCI_sort=sorted(newci.items(),key=operator.itemgetter(1))

id_NewCI = [x for x,_ in newCI_sort]
id_NewCI_value=[x for _,x in newCI_sort]






csv_data = pd.read_csv(r'C:\Users\Administrator\Desktop\3th_result\SIR_matlab\Router\router.csv')
#print csv_data
sir_node=[]
for id in range(1,11):
    index=round(float(id)/100.0,2)
    #print index
    p001= list(csv_data[str(index)])
    #print p001
    dict_p001={}
    for i in range (len(p001)):
        #print i+1
        dict_p001[i+1]=p001[i]
    #print dict_p001
    # file_dict.write(str(dict_p001)+'\n')
    dict_p001_sort=sorted(dict_p001.items(),key=operator.itemgetter(1))
    # print dict_p001_sort
    dict_p001_node= [x for x,_ in dict_p001_sort]
    dict_p001_value=[x for _,x in dict_p001_sort]
    #print dict_p001_node
    # file_node.write(str(dict_p001_node)+'\n')
    # file_value.write(str(dict_p001_value)+'\n')
    sir_node.append(dict_p001_node)
#print sir_node

'''
newCI_sort=sorted(newci.items(),key=operator.itemgetter(1),reverse=True)
#print newCI_sort
res_list_NewCI = [x for x,_ in newCI_sort]
print res_list_NewCI
file.write("time_newci"+str(time_newci)+'\n')
file.write(str(res_list_NewCI)+'\n')


'''
print len(id_between)
id_between=map(eval, id_between)
id_closeness=map(eval, id_closeness)
id_pagerank=map(eval, id_pagerank)
id_kcore=map(eval, id_kcore)
id_CI=map(eval, id_CI)
id_NewCI=map(eval, id_NewCI)

bc=[]
cc=[]
ec=[]
k_core=[]
ci=[]
newci=[]
for i in range(0,len(sir_node)):
    print  len(sir_node[i])
    print len(id_between)
    bc_value=cal_kendall_tau(id_between,sir_node[i])
    bc.append(bc_value)
    cc_value=cal_kendall_tau(id_closeness,sir_node[i])
    cc.append(cc_value)
    ec_value=cal_kendall_tau(id_pagerank,sir_node[i])
    ec.append(ec_value)
    k_core_value=cal_kendall_tau(id_kcore,sir_node[i])
    k_core.append(k_core_value)
    ci_value=cal_kendall_tau(id_CI,sir_node[i])
    ci.append(ci_value)
    newci_value=cal_kendall_tau(id_NewCI,sir_node[i])
    newci.append(newci_value)
    
print bc
file.write(str(bc)+'\n')
file.write(str(cc)+'\n')
file.write(str(ec)+'\n')
file.write(str(k_core)+'\n')
file.write(str(ci)+'\n')
file.write(str(newci)+'\n')

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
#ax.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['degree']),'k--',label='DC')
ax.plot(np.arange(10) ,np.array(bc),'c--',label='BC')
ax.plot(np.arange(10) ,np.array(cc),'y--',label='CC')
ax.plot(np.arange(10) ,np.array(ec),'m--',label='EC')
ax.plot(np.arange(10) ,np.array(k_core),'b--',label='k-core')
ax.plot(np.arange(10) ,np.array(ci),'g-',label='CI')
ax.plot(np.arange(10) ,np.array(newci),'r-',label='NewCI')
#ax.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['newci91']),'y-',label='NewCI_B/R')
#ax.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['newci19']),'k-',label='NewCI_S/R')
#ax.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['newci_s']),'c-',label='NewCI_B')
#ax.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['newci_b']),'m-',label='NewCI_S')
# ax.spines['bottom'].set_linewidth(0.5)
# ax.spines['left'].set_linewidth(0.5)
# ax.spines['right'].set_linewidth(0.5)
# ax.spines['top'].set_linewidth(0.5)
# plt.vlines(5, 0.36, 0.6, colors ="k", linestyles = "dashed")
plt.tick_params(labelsize=7.5)
# vlines(x, ymin, ymax)
# hlines(y, xmin, xmax)

#plt.xticks(x,labels,fontsize=7.5,rotation=30)
ax.legend(loc='best',ncol=3,fontsize=7.5,frameon=False)
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

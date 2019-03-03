__author__='SONG Jiaxiu'
# -*- coding: utf-8 -*-
"""
This module implements degree-ci.
"""
import networkx as nx
import matplotlib
import copy
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

matplotlib.rcParams['font.family'] = 'sans-serif'  
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
__author__ = """\n""".join(['Jordi Torrents <jtorrents@milnou.net>',
                            'Katy Bold <kbold@princeton.edu>',
                            'Aric Hagberg <aric.hagberg@gmail.com)'])

#*******************************************************************************************#
##
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
##2
def make_link(G, node1, node2):
    if node1 not in G:
        G[node1] = {}
    (G[node1])[node2] = 1
    if node2 not in G:
        G[node2] = {}
    (G[node2])[node1] = 1
    return G
def remove_node1(G,node):
    for k in G.neighbors(node):
        G.remove_edge(node,k)
    return G

def Degree_Centrality(G):
    Degree_Centrality = nx.degree_centrality(G)
    #print "Degree_Centrality:", sorted(Degree_Centrality.iteritems(), key=lambda d:d[1], reverse = True)
    return Degree_Centrality
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

def Collective_Influence1(G, l=1):
    Collective_Influence_Dic = {}
    node_set = G.nodes()
    for nid in node_set:
        CI = 0
        neighbor_set = []
        neighbor_hop_1 = G.neighbors(nid)
        

        total_reduced_degree = 0
        for id in neighbor_hop_1:
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
def New_Collective_Influence(G):
    G_CI_value=Collective_Influence(G)
    G_Balance=l_Balance(G)
    G_Strength=l_Connection_strength(G)
    G_c=nx.clustering(G)
    ##等价
    # G_Robustness={} 
    # for (x,y) in G_init.edges(): make_link(G_R,x,y)
    # #print G_R
    # for v in G_R.keys():
    #     clustering_coefficient(G_R,v), " "+v
    #     G_Robustness[v]=clustering_coefficient(G_R,v)
    newCI={}
    newCI91={}
    newCI19={}
    newCI_B={}
    newCI_S={}
    for nid in G_Balance.keys():
        #newCI[nid]=round(float((0.5*G_Balance[nid]+0.5*G_Strength[nid])*G_CI_value[nid])/float(1+G_Robustness[nid]),3)
        #newCI[nid]=round(float((0.5*G_Balance[nid]+0.5*G_Strength[nid])*G_CI_value[nid])/float(1+G_c[nid]),3)
        newCI[nid]=round(float(0.5*G_Balance[nid]/float(1+G_c[nid])*G_CI_value[nid]+0.5*G_Strength[nid]*G_CI_value[nid]),3)
        newCI91[nid]=round(float((1.0*G_Balance[nid]+0*G_Strength[nid])*G_CI_value[nid])/float(1+G_c[nid]),3)
        newCI19[nid]=round(float((0*G_Balance[nid]+1.0*G_Strength[nid])*G_CI_value[nid])/float(1+G_c[nid]),3)
        newCI_B[nid]=round(float(1.0*G_Balance[nid]*G_CI_value[nid]),3)
        newCI_S[nid]=round(float(1.0*G_Strength[nid]*G_CI_value[nid]),3)
        #print nid, newCI

    return newCI,newCI91,newCI19,newCI_B,newCI_S
    #G_NewCI=dict(0.5*G_Balance)
    #list_G_CI_value

G=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\karate1.csv")
#a=G.number_of_nodes()
G_degree=Degree_Centrality(G)
G_ci=Collective_Influence(G)
newci,newci91,newci19,newci_b,newci_s=New_Collective_Influence(G)
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels = False, node_size = 1000)
Degree_sort=sorted(G_degree.iteritems(), key=lambda d:d[1])
# print Degree_sort
# print G_degree 
# print G_ci
# print newci
# G_ci_list=[]
# for nid in G_ci:
#     #print nid
#     neighbor_hop_1 = G.neighbors(nid)
#     all_nei=list(set(neighbor_hop_1).union(set(nid)))
#     nei_ci=[]
#     nei_ci_all=[]
#     for nnid in neighbor_hop_1:
#         nei_ci.append(G_ci[nnid])
#         nei_ci_all.append(G_ci[nnid])
#     nei_ci_all.append(G_ci[nid])
#     # print nei_ci
#     # print nei_ci_all
#     max_ci=max(nei_ci_all)
#     if max_ci==G_ci[nid]:
#         G_ci_list.append(nid)
#     else:
#         pass
# print G_ci_list

# for i in G_ci_list:
#     #print i
#     G=remove_node1(G,i)
#     G.remove_node(i)

# G_ci1=Collective_Influence1(G)
# G_ci_list1=[]
# for nid in G_ci1:
#     print nid
#     neighbor_hop_11 = G.neighbors(nid)
#     #print list(set(neighbor_hop_11))
#     all_nei1=list(set(neighbor_hop_11).union(set(nid)))
#     nei_ci1=[]
#     nei_ci_all1=[]
#     nei_ci1_node=[]
#     nei_ci_all1_node=[]
#     for nnid in neighbor_hop_11:
#         #print nnid
#         nei_ci1.append(G_ci[nnid])
#         nei_ci_all1.append(G_ci[nnid])
#         nei_ci_all1_node.append(nnid)
#         nei_ci1_node.append(nnid)
#     nei_ci_all1.append(G_ci[nid])
#     nei_ci_all1_node.append(nid)
#     print nei_ci1
#     print nei_ci_all1
#     print nei_ci1_node
#     print nei_ci_all1_node
#     max_ci1=max(nei_ci_all1)
#     if max_ci1==G_ci[nid]:
#         G_ci_list1.append(nid)
#     else:
#         pass
# print G_ci_list1


id_degree = [x for x,_ in Degree_sort]
id_degree_value=[x for _,x in Degree_sort]
ci_value=[]
for i in id_degree:
    # print sir_email[int(i)]
    ci_value.append(G_ci[i])
# plt.show(

newci_value=[]
for i in id_degree:
    # print sir_email[int(i)]
    #print i
    newci_value.append(int(newci[i]))
# print ci_value
# print newci_value


figsize = 11,9

fig=plt.figure()
plt.rcParams['font.sans-serif']=['SimHei']
ax=fig.add_subplot(1,1,1)
#ax = plt.subplots(figsize=figsize)
#ax.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['degree']),'k--',label='DC')
ax.plot(np.array(id_degree_value,dtype=float) ,np.array(ci_value),'k.')
# ax.set_xlabel("DC值",fontstyle='italic')
# ax.set_ylabel("CI值",fontstyle='italic')
plt.tick_params(labelsize=7.5)
font2 = {
'weight' : 'normal',
'size'   : 30,
}
s='DC值'
s1='CI值'
# ax.set_xlabel(u'DC值',font2)
# ax.set_ylabel(s1.decode("utf-8"),font2)
ax.set_xlabel(u'DC值',fontstyle='italic',fontsize=7.5)
ax.set_ylabel(s1.decode("utf-8"),fontstyle='italic',fontsize=7.5)
# fig.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95, hspace=0.2, wspace=0.25)
fig.subplots_adjust(left=0.16, bottom=0.17, right=0.95, top=0.95, hspace=0.2, wspace=0.25)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(3,2.1)
fig.savefig(r'C:\Python27\sjxwork\NewCI_Centrality_Code\ci-dreepng.png', dpi=600)

fig.show()

# fig1=plt.figure()
# plt.rcParams['font.sans-serif']=['SimHei']
# ax1=fig1.add_subplot(1,1,1)
# #ax = plt.subplots(figsize=figsize)
# #ax.plot(np.array(remove_rate,dtype=float) ,np.array(connected_components_nums['degree']),'k--',label='DC')
# ax1.plot(np.array(id_degree_value,dtype=float) ,np.array(newci_value),'b.')
# # ax.set_xlabel("DC值",fontstyle='italic')
# # ax.set_ylabel("CI值",fontstyle='italic')
# # plt.tick_params(labelsize=23)

# s2='DC值'
# s3='NewCI值'
# # ax1.set_xlabel(u'DC值',font2)
# # ax1.set_ylabel(s3.decode("utf-8"),font2)
# # fig1.show()
# # raw_input()
# ax1.set_xlabel(u'DC值',fontstyle='italic',fontsize=7.5)
# ax1.set_ylabel(s3.decode("utf-8"),fontstyle='italic',fontsize=7.5)
# fig1.subplots_adjust(left=0.17, bottom=0.19, right=0.95, top=0.95, hspace=0.2, wspace=0.25)
# fig1 = matplotlib.pyplot.gcf()
# fig1.set_size_inches(3,2.1)

# fig1.savefig(r'C:\Python27\sjxwork\NewCI_Centrality_Code\ci-degreepng.png', dpi=600)
# fig1.show()
raw_input()

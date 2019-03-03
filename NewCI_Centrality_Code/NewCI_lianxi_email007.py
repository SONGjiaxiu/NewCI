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
from NewCI_main import MN_Collective_Influence
from NewCI_main import MN_New_Collective_Influence 
from SIR_data import sir_karate
from SIR_data import sir_email001
from SIR_data import sir_email005
from SIR_data import sir_email007

##########********************数据存放文件****************
#file = open(r"C:\Users\Administrator\Desktop\3th_result\karate.txt", "w+")
#file = open(r"C:\Users\Administrator\Desktop\3th_result\Celegans.txt", "w+")
file = open(r"C:\Python27\sjxwork\NewCI_Centrality_Code\result\email.txt", "w+")
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
        newCI[nid]=round(float((0.5*G_Balance[nid]+0.5*G_Strength[nid])*G_CI_value[nid])/float(1+G_c[nid]),3)
        #print nid, newCI
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
    Eigen_Centrality = nx.eigenvector_centrality(G)
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
#G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset\Celegans.csv")
G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset_csv\Email.csv")
#G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset_csv\Email.csv")
#G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset\Power.csv")
#G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset\Router.csv")
#G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset\Yeast.csv")
#print G_init.edges()

###度
time_degree_start=time.time()
G_degree=Degree_Centrality(G_init)
time_degree=time.time()-time_degree_start
print "degree"
print time_degree
Degree_sort=sorted(G_degree.iteritems(), key=lambda d:d[1])
print Degree_sort
##使用解包代替x[0]
# res_list_d = [x for x,_ in Degree_sort]
# print res_list_d
# file.write("degree_time"+str(time_degree)+'\n')
# file.write(str(res_list_d)+'\n')
file.write("degree"+'\n')
id_degree = [x for x,_ in Degree_sort]
id_degree_value=[x for _,x in Degree_sort]
print id_degree
print id_degree_value
file.write(str(id_degree_value)+'\n')
yingxiangli_value_degree=[]
for i in id_degree:
    # print sir_email[int(i)]
    yingxiangli_value_degree.append(sir_email007[int(i)])
# print yingxiangli_value_degree
file.write(str(yingxiangli_value_degree)+'\n')

###介数
time_Between_start=time.time()
G_Between=Between_Centrality(G_init)
time_between=time.time()-time_Between_start
print "between"
print time_between
Between_sort=sorted(G_Between.iteritems(), key=lambda d:d[1])
#print Between_sort
# res_list_b = [x for x,_ in Between_sort]
# print res_list_b
# file.write("time_between"+str(time_between)+'\n')
# file.write(str(res_list_b)+'\n')

file.write("between"+'\n')
id_between = [x for x,_ in Between_sort]
id_between_value=[x for _,x in Between_sort]
# print id_between
# print id_between_value
file.write(str(id_between_value)+'\n')
yingxiangli_value_between=[]
for i in id_between:
    # print sir_email[int(i)]
    yingxiangli_value_between.append(sir_email007[int(i)])
# print yingxiangli_value_between
file.write(str(yingxiangli_value_between)+'\n')


###接近度中心性
time_closeness_start=time.time()
G_Closeness=Closeness_Centrality(G_init)
time_closeness=time.time()-time_closeness_start
print "closeness"
print time_closeness
Closeness_sort=sorted(G_Closeness.iteritems(), key=lambda d:d[1])
#print Closeness_sort
# res_list_c = [x for x,_ in Closeness_sort]
# print res_list_c
# file.write("time_closeness"+str(time_closeness)+'\n')
# file.write(str(res_list_c)+'\n')

file.write("closeness"+'\n')
id_closeness = [x for x,_ in Closeness_sort]
id_closeness_value=[x for _,x in Closeness_sort]
# print id_closeness
# print id_closeness_value
file.write(str(id_closeness_value)+'\n')
yingxiangli_value_closeness=[]
for i in id_closeness:
    # print sir_email[int(i)]
    yingxiangli_value_closeness.append(sir_email007[int(i)])
# print yingxiangli_value_closeness
file.write(str(yingxiangli_value_closeness)+'\n')



###PageRank中心性
time_pagerank_start=time.time()
G_Page_Rank=Page_Rank(G_init)
time_pagerank=time.time()-time_pagerank_start
print "pagerank"
print time_pagerank
Page_Rank_sort=sorted(G_Page_Rank.iteritems(), key=lambda d:d[1])
#print Page_Rank_sort
# res_list_r = [x for x,_ in Page_Rank_sort]
# print res_list_r
# file.write("time_pagerank"+str(time_pagerank)+'\n')
# file.write(str(res_list_r)+'\n')

file.write("pagerank"+'\n')
id_pagerank = [x for x,_ in Page_Rank_sort]
id_pagerank_value=[x for _,x in Page_Rank_sort]
# print id_pagerank
# print id_pagerank_value
file.write(str(id_pagerank_value)+'\n')
yingxiangli_value_pagerank=[]
for i in id_pagerank:
    # print sir_email[int(i)]
    yingxiangli_value_pagerank.append(sir_email007[int(i)])
# print yingxiangli_value_pagerank
file.write(str(yingxiangli_value_pagerank)+'\n')





###k-核
time_kshell_start=time.time()
G_KShell=KShell_Centrality(G_init)
time_kshell=time.time()-time_kshell_start
# print "k_core"
# print time_kshell
KShell_sort=sorted(G_KShell.iteritems(), key=lambda d:d[1])
#print KShell_sort
# res_list_k = [x for x,_ in KShell_sort]
# print res_list_k

file.write("k-core"+'\n')
id_kcore = [x for x,_ in KShell_sort]
id_kcore_value=[x for _,x in KShell_sort]
# print id_kcore
# print id_kcore_value
file.write(str(id_kcore_value)+'\n')
yingxiangli_value_kcore=[]
for i in id_kcore:
    # print sir_email[int(i)]
    yingxiangli_value_kcore.append(sir_email007[int(i)])
# print yingxiangli_value_kcore
file.write(str(yingxiangli_value_kcore)+'\n')


# file.write("time_kshell"+str(time_kshell)+'\n')
# file.write(str(res_list_k)+'\n')

###CI
time_CI_start=time.time()
CI=Collective_Influence(G_init)
time_CI=time.time()-time_CI_start
print "ci"
print time_CI
#print a
CI_sort=sorted(CI.items(),key=operator.itemgetter(1))
#print CI_sort

file.write("ci"+'\n')
id_CI = [x for x,_ in CI_sort]
id_CI_value=[x for _,x in CI_sort]
# print id_CI
# print id_CI_value
file.write(str(id_CI_value)+'\n')
yingxiangli_value_CI=[]
for i in id_CI:
    # print sir_email[int(i)]
    yingxiangli_value_CI.append(sir_email007[int(i)])
# print yingxiangli_value_CI
file.write(str(yingxiangli_value_CI)+'\n')



# res_list_CI = [x for x,_ in CI_sort]
# print res_list_CI
# file.write("time_CI"+str(time_CI)+'\n')
# file.write(str(res_list_CI)+'\n')
B={}
B=l_Balance(G_init)
#print B
##************************计算邻域鲁棒性以被等价替代*******************************************************
# G_R = {}
# G_init_Robustness={}
# #for (x,y) in flights: make_link(G,x,y)
# for (x,y) in G_init.edges(): make_link(G_R,x,y)
# #print G_R
# for v in G_R.keys():
#     clustering_coefficient(G_R,v), " "+v
#     G_init_Robustness[v]=clustering_coefficient(G_R,v)
#     #clustering_coefficient(G_R,v), " "+v
# #print G_init_Robustness

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
# print newCI_sort

file.write("newci"+'\n')
id_NewCI = [x for x,_ in newCI_sort]
id_NewCI_value=[x for _,x in newCI_sort]
# print id_NewCI
# print id_NewCI_value
file.write(str(id_NewCI_value)+'\n')
yingxiangli_value_NewCI=[]
for i in id_NewCI:
    # print sir_email[int(i)]
    yingxiangli_value_NewCI.append(sir_email007[int(i)])
#print yingxiangli_value_NewCI
file.write(str(yingxiangli_value_NewCI)+'\n')

# res_list_NewCI = [x for x,_ in newCI_sort]
# print res_list_NewCI
#file.write("time_newci"+str(time_newci)+'\n')
#file.write(str(res_list_NewCI)+'\n')

# time_gajinnewci_start=time.time()
# newlianxi=MN_New_Collective_Influence(G_init)
# time_gaijinnewCI=time.time()-time_gajinnewci_start
# print newlianxi
# print "gaijinnewci"
# print time_gaijinnewCI
#print sir_email



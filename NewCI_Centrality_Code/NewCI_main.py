#!/usr/bin/python
# -*- coding: utf-8 -*-
import linecache
import string
import os
import math
import time
import networkx as nx
import sys
from collections import Counter
import operator

#newCICI中心性的2阶实现
def MN_Collective_Influence(G, l=2):
    Collective_Influence_Dic = {}
    l_Connection_strength_Dic={}
    l_Balance_Dic = {}
    newCI={}
    node_set = G.nodes()
    Connection_num=0
    for nid in node_set:
        degree = G.degree(nid)
        Neighbor_Set = G.neighbors(nid)
        seq_neibhood=[]
        for ngb in Neighbor_Set:
            #print ngb
            seq_neibhood.append(G.degree(ngb))
        
        degree_max=max(seq_neibhood)
        
        degree_min=min(seq_neibhood)
        
        l_Balance_Dic[nid] =round(float(1.0 / float(float(degree_max-degree_min)/(degree_max+degree_min)+1) ),3 )
        if len(Neighbor_Set)==1:
            Connection_num=1
            #print nid 
            l_Connection_strength_Dic[nid]=1.0
            #print nid,l_Connection_strength_Dic[nid]
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
            newCI[nid]=round(float((0.5*l_Connection_strength_Dic[nid]+0.5*l_Balance_Dic[nid])*Collective_Influence_Dic[nid]),3)
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
            newCI[nid]=round(float((0.5*l_Connection_strength_Dic[nid]+0.5*l_Balance_Dic[nid])*Collective_Influence_Dic[nid]),3)
    return newCI
def MN_New_Collective_Influence(G):
    G_NEWCI_value=MN_Collective_Influence(G)
   
    newCI={}
    for nid in G_NEWCI_value.keys():
        #newCI[nid]=round(float((0.5*G_Balance[nid]+0.5*G_Strength[nid])*G_CI_value[nid])/float(1+G_Robustness[nid]),3)
        newCI[nid]=round(float(G_NEWCI_value[nid])/float(1+nx.clustering(G)[nid]),3)
        #print nid, newCI
    return newCI
    #G_NewCI=dict(0.5*G_Balance)
    #list_G_CI_value
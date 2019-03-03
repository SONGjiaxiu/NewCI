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

def ave(degree):#平均度计算
    s_um = 0
    for i in range(len(degree)):
        s_um =s_um+i*degree[i]
    return round(float(s_um)/float(nx.number_of_nodes(G_init)),3)
#ave(degree)



#G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset\Email.csv")
#G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\karate1.csv")
#G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset\Celegans.csv")
#G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset\Email.csv")
#G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset\jazz.csv")
#G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset\Power.csv")
#G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset\Router.csv")
#G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset\Yeast.csv")
#G_init=createGraph(r"C:\Users\Administrator\Desktop\dataset\problem_data\problem_data\Facebook_S.csv")
G_init=createGraph(r"C:\Users\Administrator\Desktop\Grid-Yeast.csv")
#G_init=createGraph(r"C:\Users\Administrator\Desktop\CSV\CSV\Wiki.csv")
#G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset_csv\facebook.csv")
largest_cc = len(max(nx.connected_components(G_init), key=len))
print largest_cc
print 'number of nodes'
print G_init.number_of_nodes()
print 'edge of nodes'
print G_init.number_of_edges()
degree = nx.degree_histogram(G_init)
k=ave(degree)
print 'avg degree of nodes'
print k
#print degree[1]
deg={}
deg2={}
sum=0.0
for i in G_init.nodes():
    deg[i]=G_init.degree(i)
    deg2[i]=deg[i]*deg[i]
    sum+=deg2[i]
degree_max=sorted(deg.items(),key=lambda d:d[1],reverse=True)
print 'max degree'
print degree_max[0]
print 'average_clustering'
print(nx.average_clustering(G_init))
print 'average_shortest_path_length'
D=max(nx.connected_component_subgraphs(G_init), key=len)
print(nx.average_shortest_path_length(D))
#print(nx.average_shortest_path_length(G_init)) 
print 'assortativity_coefficient'
print round(nx.degree_assortativity_coefficient(G_init),3)
print "H"
H=round(sum/(float(k*k)*float(G_init.number_of_nodes())),3)
print H
print "spreading lamida"
lamida=float(k)/(float(sum)/float(G_init.number_of_nodes())-k)
print round(lamida,3)
rc = nx.rich_club_coefficient(G_init,normalized=False)#富人俱乐部
#print "rich_club"
print rc
largest_cc = len(max(nx.connected_components(G), key=len))
print largest_cc
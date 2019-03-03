__author__='SONG Jiaxiu'
# -*- coding: utf-8 -*-
"""
邻接表转化成边表.
"""
import networkx as nx

f = open(r'C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset_csv\test.txt', 'r')
file = open(r'C:\Python27\sjxwork\NewCI_Centrality_Code\Dataset_csv\test1.txt', "w+")
G = nx.read_adjlist(f, create_using=nx.Graph())
print len(G.nodes())
# for line in f.readlines():
#     #line=line.strip('\n').replace(' ','')
#     line=line.strip('\n')
#     i=line.split(' ')
#     print i
#     # for j in range(1,len(i)) :
#     #     print i[0],i[j]
#     #     file.write(str(i[0])+','+str(i[j])+','+str(1)+'\n')
for i, j in G.edges():
    file.write(str(i)+','+str(j)+','+str(1)+'\n')
f.close()
file.close()
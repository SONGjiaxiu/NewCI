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

#a=set({(1,2)})
#b=set({2,3,4})
#print a.intersection(b)

def make_link(G, node1, node2):
    if node1 not in G:
        G[node1] = {}
    (G[node1])[node2] = 1
    if node2 not in G:
        G[node2] = {}
    (G[node2])[node1] = 1
    return G
conn=[[2,3],[2,4],[5,6]]
G = nx.Graph()
#G.add_edge(2,3)
#G.add_edge(2,4)
#G.add_edge(5,6)
for k in conn:
    G.add_edge(k[0],k[1])
H=len(list(nx.connected_components(G)))
print H
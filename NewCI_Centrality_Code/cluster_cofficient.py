__author__='SONG Jiaxiu'
# node's local cluster_cofficient
import networkx as nx             
import matplotlib.pyplot as plt
from networkx.generators.atlas import *

def make_link(G, node1, node2):
    if node1 not in G:
        G[node1] = {}
    (G[node1])[node2] = 1
    if node2 not in G:
        G[node2] = {}
    (G[node2])[node1] = 1
    return G

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

G_init=createGraph(r"C:\Python27\sjxwork\NewCI_Centrality_Code\karate1.csv")
print  G_init.edges()

#for (n,nbr) in G.edges():
#    print n,nbr

flights = [("ORD", "SEA"), ("ORD", "LAX"), ('ORD', 'DFW'), ('ORD', 'PIT'),
           ('SEA', 'LAX'), ('LAX', 'DFW'), ('ATL', 'PIT'), ('ATL', 'RDU'),
           ('RDU', 'PHL'), ('PIT', 'PHL'), ('PHL', 'PVD')]

G = {}
#for (x,y) in flights: make_link(G,x,y)
for (x,y) in G_init.edges(): make_link(G,x,y)
print G

def clustering_coefficient(G,v):
    neighbors = G[v].keys()
    if len(neighbors) == 1: return 0.0
    links = 0
    for w in neighbors:
        for u in neighbors:
            if u in G[w]: links += 0.5
    return 2.0*links/(len(neighbors)*(len(neighbors)-1))

for v in G.keys():
    print clustering_coefficient(G,v), " "+v

total = 0
for v in G.keys():
    total += clustering_coefficient(G_init,v)

print total/len(G)

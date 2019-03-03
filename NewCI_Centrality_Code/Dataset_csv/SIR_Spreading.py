__author__='SONG Jiaxiu'

import snap
import numpy as np
import random
import matplotlib.pyplot as plt
import sets
from sets import Set
import scipy
from scipy import stats
from collections import Counter

iterations = 10
r_spreading={}
def SIR(Graph, I, beta, delta):
	n = Graph.GetNodes()
	S = set()
	for NI in Graph.Nodes():
		if NI.GetId() not in I:
			S.add( NI.GetId())
		
	R = set()
	
	
	while len(I)!= 0:
		# nodes no longer susceptible after current iteration
		Sprime = set()
		# newly infected nodes after current iteration
		Iprime = set()
		# nodes no longer infected after current iteration	
		Jprime = set()
		# newly recovered nodes after current iteration
		Rprime = set()

		for NI in Graph.Nodes():
			id = NI.GetId()
			if id in S:
				deg = NI.GetDeg()
				for i in range(deg):
					nbrId = NI.GetNbrNId(i)
					if nbrId in I:
						r = random.random()
						if r < beta:
							Sprime.add(id)
							Iprime.add(id)
		
			elif id in I:
				r = random.random()
				if r < delta:
					Jprime.add(id)
					Rprime.add(id)
					
		S.difference_update(Sprime)
		I.update(Iprime)
		I.difference_update(Jprime)
		R.update(Rprime)
	
	Ipercent = len(R) / (1.0* n)
	
	#print (len(R), n)
	return Ipercent


def R_Spreading(Graph):
    #r_spreading={}
    #Graph = snap.LoadEdgeList(snap.PUNGraph, "C:\Python27\sjxwork\NewCI_Centrality_Code\karate.txt", 0, 1)
    # for NI in G2.Nodes():
    #     print "node id %d with out-degree %d and in-degree %d" % (NI.GetId(), NI.GetOutDeg(), NI.GetInDeg())
    # # traverse the edges
    # for EI in G2.Edges():
    #     print "edge (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId())
    #     # traverse the edges by nodes
    # for NI in G2.Nodes():
    #     for Id in NI.GetOutEdges():
    #         print "edge (%d %d)" % (NI.GetId(), Id)
    n=Graph.GetNodes()
    
    avg=0.0
    sum=0.0
    #print n
    for nid in Graph.Nodes():
        #print nid.GetId()
        for i in range(iterations):
            I=set()
            I.add(nid.GetId())
            #I.add(nid)

            print nid.GetId()
            ipercent = SIR(Graph, I, 0.05, 1)
            sum += ipercent
        avg=round(float(float(sum)/float(100)),3)
        r_spreading[nid.GetId()]=avg
        print r_spreading[nid.GetId()]
    return r_spreading
    

Graph = snap.LoadEdgeList(snap.PUNGraph, "C:\Python27\sjxwork\NewCI_Centrality_Code\karate.txt", 0, 1)
r={}
r=R_Spreading(Graph)
print r
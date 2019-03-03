import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import time
import pickle
import random
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
#py.init_notebook_mode(connected=True)


##https://github.com/UmbertoJr/Spread-of-SIR-model-in-a-Social-Network/blob/master/Homework%201%20Umberto%20Junior%20Mele%20graph%20from%20emails.ipynb
#preprocessing dataset and creation of networkx graph
init = time.time()
dat = pd.read_csv(r'C:\Python27\sjxwork\NewCI_Centrality_Code\email-Eu-core.txt')


def run_SIR_simulation(G_e, seed, mu = 0.07, beta= 0.3):
    """G_edge is the graph, seed is the initial nodes infected, mu is the probability to recover,
    while beta is the prob to be infected"""
    S = set(G_e.nodes)
    if len(seed)==0:
        I=set([np.random.choice(G_e)])
    else:
        I= set(seed)
    S = S - I
    R=set()
    S_I =set()
    I_R = set()
    t=0
    while True:
        yield ( {'R': R, 'S':S, 'I':I, 'S->I':S_I, 'I->R':I_R , 't':t})
        if len(I)==0:
            break

        t+=1
        S_I = set()
        I_R = set()
        for inf in I:
            neig = set(G_e[inf])
            neig_S = neig.intersection(S)

            for n in neig_S:
                if np.random.uniform()<= beta:
                    S.remove(n)
                    S_I.add(n)
            if np.random.uniform()<= mu:
                R.add(inf)
                I_R.add(inf)
        I = I.union(S_I)
        I = I.difference(I_R)
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import numpy as np
import time
import os
import pandas as pd
import random
from collections import Counter
from scipy import stats
from si_animator import plot_network_usa


def SI_model(event_data,p,infected):
    #The for loop loops over events in a dataframe (thus df.iterrows() is used)
    #The for loop first checks whether the events Source airport is among the infected_airports and that the Destination airport is not amongst them
    #The for loop then checks that the StartTime of the flight is later than the time that the Source airport got infected
    #Then random number is generated and checked that it is smaller or equal to the probability of infection and decides whether to infect the destination airport
    #Then the destination airport is saved amongst the infected airports to the dictionary and to the list
    
    infected_airports = list(infected.keys())
                            
    for index, flight in event_data.iterrows():
        if flight["Source"] in infected_airports and flight["Destination"] not in infected_airports:
            if flight["StartTime"] >= infected[flight["Source"]]:
                random_n = np.random.random()
                if random_n <= p:
                    infected[flight["Destination"]] = flight["EndTime"]
                    infected_airports.append(flight["Destination"])
                    
    return infected 
    
def SI_model_immunization(event_data,p,infected,immunized):
    #The for loop loops over events in a dataframe (thus df.iterrows() is used)
    #The for loop first checks whether the events Source airport is among the infected_airports and that the Destination airport is not amongst them
    #The for loop then checks that the StartTime of the flight is later than the time that the Source airport got infected
    #Then random number is generated and checked that it is smaller or equal to the probability of infection and decides whether to infect the destination airport
    #Then the destination airport is saved amongst the infected airports to the dictionary and to the list
    #infected is list of start_nodes
    
    infected_airports = list(infected.keys())
                            
    for index, flight in event_data.iterrows():
        if flight["Source"] not in immunized:
            if flight["Source"] in infected_airports and flight["Destination"] not in infected_airports:
                if flight["StartTime"] >= infected[flight["Source"]]:
                    random_n = np.random.random()
                    if random_n <= p:
                        infected[flight["Destination"]] = flight["EndTime"]
                        infected_airports.append(flight["Destination"])
                        
    return infected 
    
def SI_model_task6(event_data,p,infected):
    
    infected_airports = list(infected.keys())
    infected_infectant = {}
                            
    for index, flight in event_data.iterrows():
        if flight["Source"] in infected_airports and flight["Destination"] not in infected_airports:
            if flight["StartTime"] >= infected[flight["Source"]]:
                random_n = np.random.random()
                if random_n <= p:
                    infected[flight["Destination"]] = flight["EndTime"]
                    infected_infectant[unicode(flight["Destination"])] = unicode(flight["Source"])
                    infected_airports.append(flight["Destination"])
                    
    return infected_infectant
    
def infection_fractions(event_data,p,infected,iterations):
    infection_lists = [[]] * iterations
    count =  0
    while count < iterations:
        infected_node =  {}
        infected_node[infected[count]] = 1229231100  
        infected_links = SI_model_task6(event_data,p,infected_node)
        infected_links = [(v, k) for k, v in infected_links.iteritems()]
        infection_lists[count] = (infected_links)
        count += 1
    infection_lists = [item for sublist in infection_lists for item in sublist]
    counter = Counter(infection_lists)
    
    return counter
    #currently saves all iteration infection lists to list of lists
    #what we need is frequencies of links in iterations (undirected) 
        

    
def prevalence_average(event_data,infected,n,p,bins,iterations,event_range):
    #Calculates the prevalence of infected node e.g. what percentage of nodes is infected in certain time point
    #Average is calculcated in each different bin (step of time) over a chosen number of iterations
    #The resulted prevalence is cumulative
    #Returns both the bin edges and the list with prevalence percentages for each bin
     
    count = 0
    prevalence_lists = [[]] * iterations 
    while count < iterations:
        freq, bins = np.histogram(SI_model(event_data,p,infected).values(), bins=bins, range=event_range)
        freq = [float(x) / n for x in freq]
        freq = np.cumsum(freq)
        prevalence_lists[count] = freq
        count += 1
        
    prevalence = np.array(prevalence_lists)
    prevalence = np.mean(prevalence, axis=0) #why has the axis=0 be here?
    
    return bins[:len(bins)-1], prevalence
    
    
def prevalence_average_immunization(event_data,infected,n,p,bins,iterations,event_range,immunized):
    #Calculates the prevalence of infected node e.g. what percentage of nodes is infected in certain time point
    #Average is calculcated in each different bin (step of time) over a chosen number of iterations
    #The resulted prevalence is cumulative
    #Returns both the bin edges and the list with prevalence percentages for each bin
     
    count = 0
    prevalence_lists = [[]] * iterations 
    while count < iterations:
        infected_node = {}
        infected_node[infected[count]] = 1229231100 
        freq, bins = np.histogram(SI_model_immunization(event_data,p,infected_node,immunized).values(), bins=bins, range=event_range)
        freq = [float(x) / n for x in freq]
        freq = np.cumsum(freq)
        prevalence_lists[count] = freq
        count += 1
    prevalence = np.array(prevalence_lists)
    prevalence = np.mean(prevalence, axis=0) #why has the axis=0 be here? 
    
    return bins[:len(bins)-1], prevalence
    
    
def select_start_nodes(immunized,iterations):
    count = 0
    start_nodes = []
    while count < iterations:
        random = np.random.randint(0,278)
        if random not in immunized:
            count += 1
            start_nodes.append(random)
            
    return start_nodes


def get_centrality_measures(network):
    """
    Calculates five centrality measures (degree, betweenness, closeness, and
    eigenvector centrality, and k-shell) for the nodes of the given network.

    Parameters
    ----------
    network: networkx.Graph()
    tol: tolerance parameter for calculating eigenvector centrality

    Returns
    --------
    [degree, betweenness, closeness, eigenvector_centrality, kshell]: list of
    numpy.arrays
    """
    
    kshell = nx.core_number(network)
    clustering = nx.clustering(network)
    degree = nx.degree_centrality(network)
    betweenness = nx.betweenness_centrality(network)
    closeness = nx.closeness_centrality(network)
    #eigenvector_centrality = nx.eigenvector_centrality(network, tol=tol).values()
    

    return [kshell, clustering, degree, betweenness, closeness]
    
    
def dict_keys_to_int(dictionary):
    new_dict =  {}
    for key,value in dictionary.items():
        new_dict[int(key)] = value
        
    return new_dict

def list_keys_to_int(a_list):
    new_list =  []
    for value in a_list:
        new_list.append(int(value))
        
    return new_list
    



if __name__ == "__main__":
    
    headers = ["Source","Destination","StartTime","EndTime","Duration"]
    event_data = pd.read_csv(r'C:\Python27\sjxwork\NewCI_Centrality_Code\SI_model-master\events_US_air_traffic_GMT.txt', sep=" ", header=None, names = headers)
    #print(event_data.head())
    airport_data = pd.read_csv(r'C:\Python27\sjxwork\NewCI_Centrality_Code\SI_model-master\US_airport_id_info.csv')
    #print(airport_data.head())
    
    event_data = event_data.sort_values(by="EndTime",ascending=1)
    #print(event_data.head())
    #print(event_data.describe())
    #print(event_data.dtypes())
    
    infected = {}
    infected[0] = 1229231100 # here you can change which airport(s) are infected in the beginnning and their infection times (now node 0 at the start of first flight)
    infected_list =  [0, 4, 41, 100, 200]
    p = 1
    n = 279
    iterations = 10
    #print(infected)
    
    bins = np.linspace(min(event_data["StartTime"]),max(event_data["EndTime"]),num=50) # you can choose the number of bins by changing the scalar
    event_range = [min(event_data["StartTime"]),max(event_data["EndTime"])]
    
    #freq, bins = np.histogram(SI_model(event_data,p,infected).values(), bins=bins, range=event_range)
    #freq = [float(x) / n for x in freq]
    #freq = np.cumsum(freq)
    #print(freq)
    #print(prevalence_average(event_data,infected,n,p,bins,iterations,event_range)[1])
    
    n = 279 # number of airports in the network
    p_infection = 0.1 # here you can change the probability of infection
    p_list = [0.01, 0.05, 0.1, 0.5, 1.0]
    
    sources = event_data["Source"]
    #print(sources.head())
    
    #print(event_data[:1])
    #print(event_data[:1]["Source"])
    
    test_events = event_data.head(1000) # since the total dataset is so large here you can choose how many rows of event data to include in your test
    #print(test_events)
    
    #task1
    '''
    print(SI_model(event_data,p_infection,infected)) #might take some time since event data is so large
    print(SI_model(test_events,p_infection,infected))
    '''
    #task2
    
    bins = np.linspace(min(event_data["StartTime"]),max(event_data["EndTime"]),num=50) # you can choose the number of bins by changing the scalar
    iterations = 10 # you can choose the number of iterations to get the average from 
    event_range = [min(event_data["StartTime"]),max(event_data["EndTime"])]
    p_list = [0.01, 0.05, 0.1, 0.5, 1.0]
    infected = {}
    infected[0] = 1229231100
    n = 279
    
    #print(p_list[0])
    
    
    values1 = prevalence_average(event_data,infected,n,p_list[0],bins,iterations,event_range)
    x1 = values1[0]
    y1 = values1[1]
    
    infected = {}
    infected[0] = 1229231100
    
    values2 = prevalence_average(event_data,infected,n,p_list[1],bins,iterations,event_range)
    x2 = values2[0]
    y2 = values2[1]
    
    infected = {}
    infected[0] = 1229231100
    
    values3 = prevalence_average(event_data,infected,n,p_list[2],bins,iterations,event_range)
    x3 = values3[0]
    y3 = values3[1]
    
    infected = {}
    infected[0] = 1229231100
    
    values4 = prevalence_average(event_data,infected,n,p_list[3],bins,iterations,event_range)
    x4 = values4[0]
    y4 = values4[1]
    
    infected = {}
    infected[0] = 1229231100
    
    values5 = prevalence_average(event_data,infected,n,p_list[4],bins,iterations,event_range)
    x5 = values5[0]
    y5 = values5[1]
    
    #print(x1)
    #print(y1)
    #print(y2)
    plt.plot(x1,y1)
    plt.plot(x2,y2)
    plt.plot(x3,y3)
    plt.plot(x4,y4)
    plt.plot(x5,y5)
    #plt.legend(['p=0.05'], loc="best")
    plt.legend(['p=0.01', 'p=0.05','p=0.1','p=0.5','p=1'], loc="best")
    plt.show()

    
   #task3
   
    '''infected_dict =  {}
    n = 279 # number of airports in the network
    p_infection = 0.1 # here you can change the probability of infection
   
   
    bins = np.linspace(min(event_data["StartTime"]),max(event_data["EndTime"]),num=50) # you can choose the number of bins by changing the scalar
    iterations = 10 # you can choose the number of iterations to get the average from 
    event_range = [min(event_data["StartTime"]),max(event_data["EndTime"])]
    
    infected_dict =  {}
    infected_dict[0] = 1229231100
    values1 = prevalence_average(event_data,infected_dict,n,p_infection,bins,iterations, event_range)
    x1 = values1[0]
    y1 = values1[1]
    
    infected_dict =  {}        
    infected_dict[4] = 1229231100    
    values2 = prevalence_average(event_data,infected_dict,n,p_infection,bins,iterations,event_range)
    x2 = values2[0]
    y2 = values2[1]
    
    infected_dict =  {}        
    infected_dict[41] = 1229231100    
    values3 = prevalence_average(event_data,infected_dict,n,p_infection,bins,iterations,event_range)
    x3 = values3[0]
    y3 = values3[1]
    
    infected_dict =  {}        
    infected_dict[100] = 1229231100    
    values4 = prevalence_average(event_data,infected_dict,n,p_infection,bins,iterations,event_range)
    x4 = values4[0]
    y4 = values4[1]
    
    infected_dict =  {}        
    infected_dict[200] = 1229231100 
    values5 = prevalence_average(event_data,infected_dict,n,p_infection,bins,iterations,event_range)
    x5 = values5[0]
    y5 = values5[1]
    
    
    plt.plot(x1,y1)
    plt.plot(x2,y2)
    plt.plot(x3,y3)
    plt.plot(x4,y4)
    plt.plot(x5,y5)
    plt.legend(['Infected node=0', 'Infected node=4','Infected node=41','Infected node=100','Infected node=200'], loc="best")
    plt.show()'''
    
    #task4
    
    '''count = 0
    iterations = 50 # Change the number of iterations
    start_nodes = []
    while count < iterations:
        count += 1
        start_nodes.append(random.randint(0,278))
    # Starting nodes are now in storage
    

    p_infection = 0.5
    count =  0
    
    
    new_dict =  {}
    key_freq = []
    for node in start_nodes:
        start_node_dict = {}
        start_node_dict[node] = 1229231100
        iteration = SI_model(event_data,p_infection,start_node_dict)
        for key, value in iteration.items():
            key_freq.append(key)
            if key in new_dict:
                new_dict[key] += value
            else:
                new_dict[key] = value
                
    key_count =  Counter(key_freq)
    for key,value in key_count.items():
        if value < iterations / 2:
            new_dict.pop('key')
    
    for key, value in new_dict.items():
        new_dict[key] =  value / key_count[key]
    
    #print(new_dict)
    
    # Now we have mean infection times for all nodes in new_dict[]
    # what we have is a dictionary containing all infected nodes and their mean infection times with nodes having less than iteration / 2 infections removed. 
    
    #Now we can read in the .edg file so the network that is provided to calculate all network properties for the network
    #Attention! The network is weighted!
    #The aggregated weighted network aggregated_US_air_traffic_network_undir.edg is constructed
    #based on the event data, so that weight of each link corresponds to the number of
    #flights between the nodes. 
    
    network_path = 'C:/Users/kette/Complex networks/aggregated_US_air_traffic_network_undir.edg'
    network = nx.read_weighted_edgelist(network_path)
    
    #print(get_centrality_measures(network)[0].keys())
    #print(get_centrality_measures(network)[0].values())
    
    #converting centrality unicode key values to int
    centrality_dict = {}
    centrality = get_centrality_measures(network)[0]
    for key,value in centrality.items():
        centrality_dict[int(key)] = value
    #print(centrality_dict)
    
    #deleting keys from centrality dict that are not in new_dict
    for key in centrality_dict.keys():
        if key not in new_dict.keys():
            centrality_dict.pop('key')
    
    x = centrality_dict.values()
    y = new_dict.values()
    print(stats.spearmanr(centrality_dict.values(),new_dict.values()))'''
    
    
    #task5
    
    '''p = 0.5
    iterations = 20
    np.random.seed(40)
    n = 279 # number of airports in the network
    
    network_path = 'C:/Users/kette/Complex networks/aggregated_US_air_traffic_network_undir.edg'
    network = nx.read_weighted_edgelist(network_path)

    #Selection of random immunized nodes
    
    count = 0
    number_of_immunized = 10 # Change the number of immunized
    random_immunized = []
    while count < number_of_immunized:
        random = np.random.randint(0,278)
        if random not in random_immunized:
            count += 1
            random_immunized.append(random)
    
    
    #selection of social random nodes
    
    count = 0
    number_of_immunized = 10 # Change the number of immunized
    social_immunized = []
    while count < number_of_immunized:
        random = np.random.randint(0,278)
        random = int(network.neighbors(str(random))[0])
        if random not in social_immunized:
            count += 1
            social_immunized.append(random)
    
    
    
    #selection of centrality measure nodes
    
    kshell = nx.core_number(network)
    strength = nx.degree(network, weight='weight')
    clustering = nx.clustering(network)
    degree = nx.degree_centrality(network)
    betweenness = nx.betweenness_centrality(network)
    closeness = nx.closeness_centrality(network)
    
    kshell_immune =  list_keys_to_int(sorted(kshell, key=kshell.get, reverse=True)[:10])
    strength_immune = list_keys_to_int(sorted(strength, key=strength.get, reverse=True)[:10])
    clustering_immune = list_keys_to_int(sorted(clustering, key=clustering.get, reverse=True)[:10])
    degree_immune = list_keys_to_int(sorted(degree, key=degree.get, reverse=True)[:10])
    betweenness_immune = list_keys_to_int(sorted(betweenness, key=betweenness.get, reverse=True)[:10])
    closeness_immune = list_keys_to_int(sorted(closeness, key=closeness.get, reverse=True)[:10])
    
    # immunized nodes are now in storage
    # adding immunized nodes to same list
    immunized = random_immunized + social_immunized + kshell_immune + strength_immune + clustering_immune + degree_immune + betweenness_immune + closeness_immune 
    #print(immunized)
    #print(kshell_immune)
    #print(clustering_immune)
    #print(degree_immune)
    
    #Selection of start nodes
    
    count = 0
    number_of_seeds = 20 # Change the number of immunized
    start_nodes = []
    while count < number_of_seeds:
        random = np.random.randint(0,278)
        if random not in immunized:
            count += 1
            start_nodes.append(random)
    
    print(start_nodes)
    
    #Average prevalence times over the 20 iterations
    bins = np.linspace(min(event_data["StartTime"]),max(event_data["EndTime"]),num=50) # you can choose the number of bins by changing the scalar
    iterations = 20 # you can choose the number of iterations to get the average from 
    event_range = [min(event_data["StartTime"]),max(event_data["EndTime"])]
    
    values1 = prevalence_average_immunization(event_data,start_nodes,n,p,bins,iterations,event_range,random_immunized)
    x1 = values1[0]
    y1 = values1[1]
    #plt.plot(x1,y1,label="Random nodes")
    #plt.legend(loc="best")
    #plt.show()
    
    values2 = prevalence_average_immunization(event_data,start_nodes,n,p,bins,iterations,event_range,social_immunized)
    x2 = values2[0]
    y2 = values2[1]
    #plt.plot(x2,y2,label="Social immunized")
    #plt.legend(loc="best")
    #plt.show()
    
    values3 = prevalence_average_immunization(event_data,start_nodes,n,p,bins,iterations,event_range,kshell_immune)
    x3 = values3[0]
    y3 = values3[1]
    #plt.plot(x3,y3,label="Kshell immunized")
    #plt.legend(loc="best")
    #plt.show()
    
    values4 = prevalence_average_immunization(event_data,start_nodes,n,p,bins,iterations,event_range,clustering_immune)
    x4 = values4[0]
    y4 = values4[1]
    #plt.plot(x4,y4,label="Clustering immunized")
    #plt.legend(loc="best")
    #plt.show()
    
    values5 = prevalence_average_immunization(event_data,start_nodes,n,p,bins,iterations,event_range,degree_immune)
    x5 = values5[0]
    y5 = values5[1]
    #plt.plot(x5,y5,label="degree immunized")
    #plt.legend(loc="best")
    #plt.show()
    
    values6 = prevalence_average_immunization(event_data,start_nodes,n,p,bins,iterations,event_range,betweenness_immune)
    x6 = values6[0]
    y6 = values6[1]
    #plt.plot(x6,y6,label="betweenness immunized")
    #plt.legend(loc="best")
    #plt.show()
    
    values7 = prevalence_average_immunization(event_data,start_nodes,n,p,bins,iterations,event_range,closeness_immune)
    x7 = values7[0]
    y7 = values7[1]
    #plt.plot(x7,y7,label="Closeness immunized")
    #plt.legend(loc="best")
    #plt.show()    
    
    values8 = prevalence_average_immunization(event_data,start_nodes,n,p,bins,iterations,event_range,strength_immune)
    x8 = values8[0]
    y8 = values8[1]
    
    
    plt.plot(x1,y1)
    plt.plot(x2,y2)
    plt.plot(x3,y3)
    plt.plot(x4,y4)
    plt.plot(x5,y5)
    plt.plot(x6,y6)
    plt.plot(x7,y7)
    plt.plot(x8,y8)
    plt.legend(['Random immunized', 'Social immunized','Kshell immunized','Clustering immunized','Degree immunized','Betweenness immunized','Closeness immunized', 'Strength immunized'], loc="best")
    plt.show()'''
    
    #task6
    
    #So adapt your code to recording the (undirected) links which are used to transmit the
    #disease. This is best done by storing for each node where it obtained the infection. Run 20
    #simulations using random nodes as seeds and p = 0.5. For each simulation, record which links
    #are used to infect yet uninfected airports
    
    # p = 0.5
    # iterations = 20

    # #Assigning random starting nodes
    # count = 0
    # number_of_seeds = iterations # Change the number of immunized
    # start_nodes = []
    # while count < number_of_seeds:
    #     random = np.random.randint(0,278)
    #     if random not in start_nodes:
    #         count += 1
    #         start_nodes.append(random)
    
    # #reading in the network
    # network_path =r'C:\Python27\sjxwork\NewCI_Centrality_Code\SI_model-master\aggregated_US_air_traffic_network_undir.edg'
    # network = nx.read_weighted_edgelist(network_path)
    
    # #Getting a dictionary with node xycoordinates 
    # xycoords = {}
    # for index, row in airport_data.iterrows():
    #     xycoords[unicode(row["id"])] = [row["xcoordviz"], row["ycoordviz"]]
        
    
    # #Getting the edges and linewidths
    # infection_edges = infection_fractions(event_data,p,start_nodes,iterations) #infection_fractions needs fixing
    # edges = infection_edges.keys()
    # print(edges)
    # linewidths = []
    # count = 0
    # for i in infection_edges.values():
    #     linewidths.append(float(i) / iterations)
    #     count += 1
        
    # #Plotting the network
    # plot_network_usa(network,xycoords,edges=edges,linewidths=linewidths)
    # plt.show()
    
    #Potting the maximum spanning tree
    '''minimum_spanning_tree = nx.minimum_spanning_tree(network,weight="weight")
    net_copy = network.copy()
    for a,b,data in net_copy.edges(data=True):
        data['weight'] = data['weight']*-1
    maximum_spanning_tree = nx.minimum_spanning_tree(net_copy,weight="weight")
    plot_network_usa(maximum_spanning_tree, xycoords)'''
    
    
    #A lot of things in this task needs to be in unicode, thus the code is a bit messy
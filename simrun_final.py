# Copyright: Team Kaigoo
import sim
import json
import sys
import networkx as nx
import heapq as hq
import random as rd
import numpy as np
import operator
import community
from collections import defaultdict
# import time
# import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import manifold

#Input: 	graph: adjacency graph
			#num_player: number of players in the game
			#num_node: how many nodes can we choose initially
    #Output:	strategy: dictionary of strategies


def generate_strategy(graph, num_player, num_node, f_name, R, pos):

    if f_name == 's1':
        s = TopNDegree(graph, num_node)
    if f_name == 's2':
        s = TopNCloseness(graph, num_node)
    if f_name == 's3':
        s = rand_TopNDegree(graph, num_node, 1.3)
    if f_name == 's4':
        s = rand_TopNCloseness(graph, num_node, 1.3)
    if f_name == 's5':
        s = Partition_Closeness(graph, num_node, 1.3, 0.05)
    if f_name == 's6':
        s = Partition_Closeness_Single(graph, num_node, 1.3, R)
    if f_name == 's7':
        s = spectral_partition_closeness(graph, 2, num_node)
    if f_name == 's8':
        s = rand_TopNMixed(graph, num_node, 1.3)
    if f_name == 's9':
        s = eyeballing(graph, pos, num_node, 1.3)

    return s
#%% strategies

def eyeballing(graph, pos, N, X):
    list_of_nodes = spectral_region(graph, pos)
    return partition_interface(graph, list_of_nodes, N, X)


def spectral_region(graph,pos):
    G = nx.Graph(graph)
    node_map = {}
    node_imap = {}
    node_list = list(G.nodes())
    for i in range(len(node_list)):
        node_map[node_list[i]] = i 
        node_imap[i] = node_list[i]  
    A = nx.adjacency_matrix(G)
    B = A.todense()
    # ## embedding
    embedding = manifold.SpectralEmbedding(n_components= 2 , affinity = 'precomputed')
    # B = A.todense()
    embedding.fit(B)
    C = embedding.fit_transform(B)
    x1, x2, y1, y2 = pos
    list_of_nodes = []
    for i in range(len(C)):
        if x1 < C[i,0] < x2 and y1 < C[i,1] < y2:
            list_of_nodes.append(node_imap[i])
    return list_of_nodes

def partition_interface(graph, list_of_nodes, N, X):
    print(list_of_nodes)
    G = nx.Graph(graph)
    close_cen = {}
    deg_cen = {}
    for node in list_of_nodes:
        close_cen[node] = nx.algorithms.closeness_centrality(G, u=node)
        deg_cen[node] = G.degree(node)
    # print(deg_cen)
    # print(close_cen)
    deg_sorted = sorted(deg_cen.items(), key=operator.itemgetter(1))
    close_sorted = sorted(close_cen.items(), key=operator.itemgetter(1))
    scores = defaultdict(int)
    for i in range(len(list_of_nodes)):
        scores[deg_sorted[i][0]] += i 
        scores[close_sorted[i][0]] += i 
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    res = []
    Ing = int(0.1*len(list_of_nodes))
    #Ing = 10
    for i in np.arange(Ing,Ing+int(N*X)):
        res.append(sorted_scores[i][0])

    nodes_list = []
    for i in range(50):
        nodes_list.append(rd.sample(res, N))
    return nodes_list

def rand_TopNDegree(graph, N, X):
    num = int(N * X)
    topNX = TopNDegree(graph, num)
    nodes_list = []
    for i in range(50):
        nodes_list.append(rd.sample(topNX, N))
    return nodes_list


def rand_TopNCloseness(graph, N, X):
    num = int(N * X)
    topNX = TopNCloseness(graph, num)
    nodes_list = []
    for i in range(50):
        nodes_list.append(rd.sample(topNX, N))
    return nodes_list


def TopNDegree(graph, N):
    G = nx.Graph(graph)
    deg_cen = nx.algorithms.degree_centrality(G)
    nodes = []
    degrees = []
    res = []
    for n in deg_cen:
        nodes.append(n)
        degrees.append(deg_cen[n])
    idx = hq.nlargest(N, range(len(degrees)), degrees.__getitem__)
    for i in idx:
        res.append(nodes[i])
    return res


def TopNCloseness(graph, N):
    G = nx.Graph(graph)
    close_cen = nx.algorithms.closeness_centrality(G)
    nodes = []
    centralities = []
    res = []
    for n in close_cen:
        nodes.append(n)
        centralities.append(close_cen[n])
    idx = hq.nlargest(N, range(len(centralities)), centralities.__getitem__)
    for i in idx:
        res.append(nodes[i])
    return res


def TopNMixed(graph, N):
    G = nx.Graph(graph)
    deg_cen = nx.algorithms.degree_centrality(G)
    close_cen = nx.algorithms.closeness_centrality(G)
    deg_sorted = sorted(deg_cen.items(), key=operator.itemgetter(1))
    close_sorted = sorted(close_cen.items(), key=operator.itemgetter(1))
    scores = defaultdict(int)
    for i in range(len(graph)):
        scores[deg_sorted[i][0]] += i 
        scores[close_sorted[i][0]] += i 
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    res = []
    #Ing = int(0.01*len(graph))
    for i in np.arange(Ing,Ing+N):
        res.append(sorted_scores[i][0])
    return res


def rand_TopNMixed(graph, N, X):
    num = int(N * X)
    topNX = TopNMixed(graph, num)
    nodes_list = []
    for i in range(50):
        nodes_list.append(rd.sample(topNX, N))
    return nodes_list


def Partition_Closeness(graph, N, X, threshold):
    G = nx.Graph(graph)
    partition = community.best_partition(G)

    nodes = defaultdict(list)
    for node, cluster in partition.items():
        nodes[cluster].append(node)

    lens_of_partition = []
    for key in nodes:
        if len(nodes[key]) >= threshold*len(G):
            lens_of_partition.append(len(nodes[key]))
    
    nodes_list = []
    for key in nodes:
        if len(nodes[key]) >= threshold*len(G):
            sub_G = G.subgraph(nodes[key])
            n = N*X*len(nodes[key])/np.sum(lens_of_partition)
            print(int(round(n)))
            topnX = TopNCloseness(sub_G, int(round(n)))
            print(topnX)
            nodes_list = nodes_list + topnX

    final_nodes_list = []
    for i in range(50):
        final_nodes_list.append(rd.sample(nodes_list, N))
    
    return final_nodes_list


def Partition_Closeness_Single(graph, N, X, R):
    G = nx.Graph(graph)
    partition = community.best_partition(G, resolution = R)

    nodes = defaultdict(list)
    for node, cluster in partition.items():
        nodes[cluster].append(node)

    max_key = 0
    max_len = 0
    for key in nodes:
        if len(nodes[key]) >= max_len:
            max_key = key
            max_len = len(nodes[key])
    
    sub_G = G.subgraph(nodes[max_key])

    return rand_TopNCloseness(sub_G, N, 1.3)
    #return rand_TopNCloseness(sub_G, N, X)

def spectral_partition(graph, n_cluster):
    G = nx.Graph(graph)
    node_map = {}
    node_imap = {}
    node_list = list(G.nodes())
    for i in range(len(node_list)):
        node_map[node_list[i]] = i 
        node_imap[i] = node_list[i]
        
    A = nx.adjacency_matrix(G)

    ## embedding

    embedding = manifold.SpectralEmbedding(n_components= 2 , affinity = 'precomputed')
    B = A.todense()
    embedding.fit(B)
    C = embedding.fit_transform(B)


    for i in range(len(C)):
        for j in range(len(C[0])):
            C[i][j] = np.exp(C[i][j])
        
    ## clustering
    nodes = defaultdict(list)
    spectral = cluster.SpectralClustering(degree = 10, n_clusters = n_cluster, affinity = 'nearest_neighbors') #n_clusters
    spectral.fit(C)
    clusters = spectral.fit_predict(C)
    for group, i in zip(clusters, range(len(clusters))):
        node = node_imap[i]
        nodes[group].append(node)
    sub_G = {}
    for i in nodes:
        sub_G[i] = G.subgraph(nodes[i])
    return sub_G


def spectral_partition_closeness(graph, n_cluster, N):
    M = int(1.5 * N)
    sub_G = spectral_partition(graph, 2)
    max_len = 0
    for i,g in sub_G.items():
        if len(g) > max_len:
            max_sub_G = g
            max_len = len(g)
    print(len(max_sub_G))

    list_graph = list(nx.connected_component_subgraphs(max_sub_G))
    for g in list_graph:
        if len(g) > len(max_sub_G)/2:
            max_sub_G_connected = max_sub_G.subgraph(g)
    print(len(max_sub_G_connected))
    if len(max_sub_G_connected) < 5000:
        nodes_list = TopNMixed(max_sub_G_connected, M)
        final_nodes_list = []
        for i in range(50):
            final_nodes_list.append(rd.sample(nodes_list, N))
        return final_nodes_list
    else:
        sub_G = spectral_partition(max_sub_G_connected,2)
        n = {}
        max_len = 0
        max_i = 0
        for i, g in sub_G.items():
            n[i] = int(len(g) * M / len(max_sub_G_connected))
            if n[i] > max_len:
                max_i = i
        n[max_i] += M - sum(n.values())
        list_of_nodes = []
        for i, g in sub_G.items():
            list_of_nodes += TopNMixed(g,n[i])
        print(len(list_of_nodes))
        final_nodes_list = []
        for i in range(50):
            final_nodes_list.append(rd.sample(nodes_list, N))
        return final_nodes_list


def output_nodes(list_of_nodes, out_path):
    with open(out_path, "w") as text_file:
        for nodes in list_of_nodes:
            for node in nodes:
                text_file.write(node + '\n')


def replay(graph, filename, list_of_nodes):
    strategies = json.load(open(filename))
    Ranks = 0

    for i in range(50):
        print('***** Round ' + str(i) + ' *****')
        # print(list_of_nodes)
        strategy = defaultdict(list)
        for key in strategies:
            strategy[key] = strategies[key][i]

        strategy['Kaigoo'] = list_of_nodes[i]

        results = sim.run(graph, strategy)
        print(results)

        scores = []
        for key in results:
            scores.append(results[key])
        scores = sorted(scores)
        score = results['Kaigoo']
        for i in range(num_player):
            if scores[i] == score:
                rank = num_player - i
        print('Rank: ' + str(rank))
        Ranks = Ranks + rank

    print('Final Rank: ' + str(Ranks/50))


if __name__ == "__main__":
    
    if len(sys.argv) != 10:
        print("usage: python simrun.py <path_to_jsonfile> <# players> <# nodes> \
            <path_to_output> <resolution> <pos_Xlow> <pos_Xhigh> <pos_Ylow> <pos_Yhigh>")
        sys.exit(1)

    # initialize
    filename = sys.argv[1]
    num_player = int(sys.argv[2])
    num_node = int(sys.argv[3])
    out_path = sys.argv[4]
    R = float(sys.argv[5])
    pos = []
    pos.append(float(sys.argv[6]))
    pos.append(float(sys.argv[7]))
    pos.append(float(sys.argv[8]))
    pos.append(float(sys.argv[9]))

    # pre-process graph into one connected component
    init_graph = json.load(open(filename))
    big_graph = nx.Graph(init_graph)
    list_graph = list(nx.connected_component_subgraphs(big_graph))
    for g in list_graph:
        if len(g) > len(big_graph)/2:
            graph = big_graph.subgraph(g)

    # generate strategy and output
    print(len(graph))
    list_of_nodes = generate_strategy(graph, num_player, num_node, 's9', R, pos)
    # print(list_of_nodes)
    output_nodes(list_of_nodes, out_path)
    #filename = '27.10.1-Kaigoo.json'
    #replay(init_graph, filename, list_of_nodes)
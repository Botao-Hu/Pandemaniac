# Copyright: Team Kaigoo
import sim
import json
import sys
import networkx as nx
import heapq as hq
import random as rd

#Input: 	graph: adjacency graph
			#num_player: number of players in the game
			#num_node: how many nodes can we choose initially
    #Output:	strategy: dictionary of strategies


def generate_strategy(graph, num_player, num_node):

    s1 = TopNDegree(graph, num_node)
    s2 = TopNCloseness(graph, num_node)
    s3 = rand_TopNDegree(graph, num_node, 2)
    s4 = rand_TopNCloseness(graph, num_node, 2)
    # print(s3,s4)
    # strategy = {"strategy1": s3, "strategy2": s4}
    return s4
#%% strategies

def rand_TopNDegree(graph, N, X):
    num = N * X
    topNX = TopNDegree(graph, num)
    nodes_list = []
    for i in range(50):
        nodes_list.append(rd.sample(topNX, N))
    return nodes_list

def rand_TopNCloseness(graph, N, X):
    num = N * X
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
    idx = hq.nlargest(N,range(len(centralities)), centralities.__getitem__)
    for i in idx:
        res.append(nodes[i])
    return res

def TopNFlowBetweeness(graph, N):
    G = nx.Graph(graph)
    betw_cen = nx.algorithms.centrality.approximate_current_flow_betweenness_centrality(G)
    nodes = []
    betweeness = []
    res = []
    for n in betweeness:
        nodes.append(n)
        centralities.append(betw_cen[n])
    idx = hq.nlargest(N,range(len(betweeness)), betweeness.__getitem__)
    for i in idx:
        res.append(nodes[i])
    return res

def output_nodes(list_of_nodes, out_path):
    with open(out_path, "w") as text_file:
        for nodes in list_of_nodes:
            for node in nodes:
                text_file.write(node + '\n')

if __name__ == "__main__":
    
    if len(sys.argv) != 5:
        print("usage: python simrun.py <path_to_jsonfile> <# players> <# nodes> <path_to_output>")
        sys.exit(1)

    # initialize
    filename = sys.argv[1]
    num_player = int(sys.argv[2])
    num_node = int(sys.argv[3])
    out_path = sys.argv[4]
    graph = json.load(open(filename))

    # generate strategy and output
    list_of_nodes = generate_strategy(graph, num_player, num_node)
    output_nodes(list_of_nodes, out_path)

    # local testing
    # results = sim.run(graph, strategy)
    # print(results)
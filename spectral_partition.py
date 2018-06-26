def spectral_partition_closeness(graph, n_cluster):
    G = nx.Graph(graph)
    #find the largest connected component G_sub
    graphs = list(nx.connected_component_subgraphs(G))
    G_sub = G.subgraph(graphs[0])
    # node_map: nodeID -> idx in A
    # node_imap: idx in A -> nodeID
    node_map = {}
    node_imap = {}
    node_list = G_sub.nodes()
    for i in range(len(node_list)):
        node_map[node_list[i]] = i 
        node_imap[i] = node_list[i]
        
    A = nx.adjacency_matrix(G_sub)

    ## embedding

    embedding = manifold.SpectralEmbedding(n_components= 2 , affinity = 'precomputed')
    B = A.todense()
    embedding.fit(B)
    C = embedding.fit_transform(B)
    ## clustering

    spectral = cluster.SpectralClustering(degree = 10,n_clusters = n_cluster, affinity = 'nearest_neighbors') #n_clusters
    spectral.fit(C)
    clusters = spectral.fit_predict(C)
    nodes = defaultdict(list)
    partition = defaultdict(int)
    for group, i in zip(clusters, range(len(clusters))):
        node = node_imap[i]
        nodes[group].append(node)
        partition[node] = group
    max_len = 0
    for i in range(n_cluster):
        if len(nodes[i])> max_len
        max_cluster = nodes[i]
    max_sub_G = G_sub.subgraph(max_cluster)
    return max_sub_G
import networkx as nx
import numpy as np

def diffuse(graph: nx.Graph,features:np.array) -> np.array:
    new_features = np.zeros(np.shape(features))
    for node,neigh_dict in graph.adjacency():
        for neigh,attr in neigh_dict.items():
            new_features[node]+= features[neigh]/graph.degree[node]
    # print(new_features)
    return new_features

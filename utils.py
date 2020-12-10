import networkx as nx
import numpy as np

def diffuse(graph: nx.Graph,features:np.array) -> np.array:
    new_features = np.zeros(np.shape(features))
    for node in graph.nodes():
        for neigh in graph.neighbors(node):
            new_features[node] += (features[neigh]/graph.degree[neigh])
    return new_features

import networkx as nx
import utils
from matplotlib import pyplot as plt
import os
import numpy as np

output_dir = ".output"
graph: nx.Graph = nx.grid_graph((10, 10))
# graph=nx.Graph(mode="slice",timerepresentation="timestamp" ,timestamp="1")
features = np.zeros((10, 10))
features[0, 0] = 1
features = features/np.sum(features)

history = nx.Graph()

for node in graph:
    graph.add_edge(node,node)

# my_layout = nx.kamada_kawai_layout(G=graph)

for i in range(1000):
    features = utils.diffuse(graph=graph, features=features)
    # print(i)
    if i % 10 == 0:
        for node in graph:
            x, y = node
            history.add_node(
                (node, i), pr=np.log(features[node]), start=i, end=i+10, viz={"position": {"x": 100*x, "y": 100*y}})
        for edge in graph.edges.data("weight"):
            # print(edge)
            # break
            u,v,_=edge
            history.add_edge((u, i), (v, i), weight=None)
        # nx.write_gexf(graph, os.path.join(output_dir, str(i)+".gexf"),version="1.2draft")
nx.write_gexf(history, os.path.join(
    output_dir, "graph.gexf"), version="1.2draft")

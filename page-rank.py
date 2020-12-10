import networkx as nx
import utils
from matplotlib import pyplot as plt
import os
import numpy as np
output_dir = ".output"

# graph: nx.Graph = nx.petersen_graph()
# graph: nx.Graph = nx.karate_club_graph()
graph: nx.Graph = nx.grid_graph((10,10))
for node in graph.nodes:
    graph.add_edge(node,node)
features = np.random.rand(10,10)
features[:,:] = 0.0001
features[0,0] = 1.0
# features = np.random.rand(nx.number_of_nodes(graph))
features = features/np.sum(features)
my_layout = nx.kamada_kawai_layout(graph)
for i in range(1000):
    features = utils.diffuse(graph, features)
    if(i%10==0):
        fig = plt.figure(1, figsize=(6, 6))
        # nx.draw(G=graph,pos=my_layout,node_color=colors,vmin=0.0,vmax=1.0)
        # print(features)
        nx.draw(G=graph, pos=my_layout, node_color=np.log(features))
        # plt.show()
        plt.savefig(os.path.join(output_dir, str(i)+".png"))
        plt.close()
    # break

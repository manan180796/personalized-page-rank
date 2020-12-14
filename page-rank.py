# from typing import Type
import csv
import io
import os
import urllib.request as urllib
import zipfile

import networkx as nx
import numpy as np

import utils
epsilon = 1e-9


# url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"

# sock = urllib.urlopen(url)  # open URL
# s = io.BytesIO(sock.read())  # read into BytesIO "file"
# sock.close()

# zf = zipfile.ZipFile(s)  # zipfile object
# txt = zf.read("football.txt").decode()  # read info file
# gml = zf.read("football.gml").decode()  # read gml data
# # throw away bogus first line with # from mejn files
# gml = gml.split("\n")[1:]
# graph = nx.parse_gml(gml)  # parse gml data


graph: nx.Graph = nx.karate_club_graph()

utils.enumerate_nodes(graph=graph)


features = np.zeros(graph.number_of_nodes())
features[0] = 1
features = features/np.sum(features)

alphas = [0.6, 0.7, 0.85, 0.9, 0.95, 1.0]
teleport = np.copy(features)


for node in graph:
    # graph.add_edge(node, node)
    pass


pr_features = {a: np.copy(features) for a in alphas}
# ppr_features = np.copy(features)

diffuzer = {
    key: utils.TeleportDiffusion(
        diffuzer=utils.NormalizedDiffusion(),
        teleport_vector=teleport,
        tele_prob=key
    )
    for key in alphas
}


for i in range(100):
    for key, cur_features in pr_features.items():
        pr_features[key] = diffuzer[key](graph=graph, features=cur_features)

result = {"original": features}

for key, cur_features in pr_features.items():
    result[f"pr(alpha={key})"] = cur_features
    result[f"pr(alpha={key}).log"] = np.log(cur_features+epsilon)

utils.set_numpy_attribute(
    graph=graph,
    attr=result
)
nx.write_gexf(graph, ".output/karate_club_graph.gexf")

import networkx as nx
import numpy as np
from typing import List, Dict
import os
import csv


class Diffusion:
    def __call__(self, graph: nx.Graph, features: np.array) -> np.array:
        new_features = np.zeros(np.shape(features))
        for node, neigh_dict in graph.adjacency():
            u = graph.nodes[node]["index"]
            for neigh in neigh_dict:
                v = graph.nodes[neigh]["index"]
                new_features[v] += features[u]/graph.degree[node]
        return new_features


class NormalizedDiffusion(Diffusion):
    def __call__(self, graph: nx.Graph, features: np.array) -> np.array:
        new_features = np.zeros(np.shape(features))
        for node, neigh_dict in graph.adjacency():
            u = graph.nodes[node]["index"]
            for neigh in neigh_dict:
                v = graph.nodes[neigh]["index"]
                new_features[v] += features[u] / \
                    np.sqrt(graph.degree[node]*graph.degree[neigh])
        return new_features


class TeleportDiffusion(Diffusion):
    def __init__(self, diffuzer: Diffusion, teleport_vector: np.array, tele_prob: float = 0.0) -> None:
        self.diffuzer = diffuzer
        self.teleport_vector = teleport_vector
        self.tele_prob = tele_prob

    def __call__(self, graph: nx.Graph, features: np.array) -> np.array:
        new_features = self.diffuzer(graph=graph, features=features)
        new_features = (1-self.tele_prob)*new_features + \
            self.tele_prob*self.teleport_vector
        return new_features


def enumerate_nodes(graph: nx.Graph):
    for i, node in enumerate(graph):
        graph.nodes[node]["index"] = i


def set_numpy_attribute(graph: nx.Graph, attr: Dict[str, np.array]):
    for node in graph:
        for key, value in attr.items():
            graph.nodes[node][key] = value[graph.nodes[node]["index"]]

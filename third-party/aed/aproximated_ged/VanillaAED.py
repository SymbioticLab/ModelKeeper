#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    VanillaAED.py

    Riesen, Kaspar, and Horst Bunke. "Approximate graph edit distance computation by means of bipartite graph matching."
    Image and Vision computing 27.7 (2009): 950-959.

    Basic implementation of edit cost operations.
"""

import glob
import itertools
import os
from itertools import chain

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

from .AproximatedEditDistance import AproximatedEditDistance
from .Plotter import plot_assignment

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"


class VanillaAED(AproximatedEditDistance):
    """
        Vanilla Aproximated Edit distance, implements basic costs for substitution insertion and deletion.
    """

    def __init__(self, del_node = 0.5, ins_node = 0.5, del_edge = 0.25, ins_edge = 0.25, metric = "euclidean"):
        self.del_node = del_node
        self.ins_node = ins_node
        self.del_edge = del_edge
        self.ins_edge = ins_edge
        self.metric = metric

    """
        Node edit operations
    """
    def node_substitution(self, g1, g2):
        """
            Node substitution costs
            :param g1, g2: Graphs whose nodes are being substituted
            :return: Matrix with the substitution costs
        """
        values1 = [v for k, v in g1.nodes(data=True)]
        v1 = [list(chain.from_iterable(l.values())) for l in values1]

        values2 = [v for k, v in g2.nodes(data=True)]
        v2 = [list(chain.from_iterable(l.values())) for l in values2]

        node_dist = cdist(np.array(v1), np.array(v2), metric=self.metric)

        return node_dist

    def node_insertion(self, g):
        """
            Node Insertion costs
            :param g: Graphs whose nodes are being inserted
            :return: List with the insertion costs
        """
        values = [v for k, v in g.nodes(data=True)]
        return [self.ins_node]*len(values)

    def node_deletion(self, g):
        """
            Node Deletion costs
            :param g: Graphs whose nodes are being deleted
            :return: List with the deletion costs
        """
        values = [v for k, v in g.nodes(data=True)]
        return [self.del_node] * len(values)

    """
        Edge edit operations
    """
    def edge_substitution(self, g1, g2):
        """
            Edge Substitution costs
            :param g1, g2: Adjacency list for particular nodes.
            :return: List of edge deletion costs
        """
        #print([list(l.values()) for l in g1], [list(l.values()) for l in g2], "=====")
        edge_dist = cdist(np.array([list(l.values()) for l in g1]), np.array([list(l.values()) for l in g2]), metric=self.metric)
        return edge_dist

    def edge_insertion(self, g):
        """
            Edge insertion costs
            :param g: Adjacency list.
            :return: List of edge insertion costs
        """
        insert_edges = [len(e) for e in g]
        return np.array([self.ins_edge] * len(insert_edges)) * insert_edges

    def edge_deletion(self, g):
        """
            Edge Deletion costs
            :param g: Adjacency list.
            :return: List of edge deletion costs
        """
        delete_edges = [len(e) for e in g]
        return np.array([self.del_edge] * len(delete_edges)) * delete_edges

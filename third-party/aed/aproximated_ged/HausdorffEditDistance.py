#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    HausdorffEditDistance.py

    Fischer, Andreas, et al. "Approximation of graph edit distance based on Hausdorff matching."
    Pattern recognition 48.2 (2015): 331-343.
"""

from .GraphEditDistance import GraphEditDistance

from scipy.optimize import linear_sum_assignment
import numpy as np

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"


class HausdorffEditDistance(GraphEditDistance):
    """
        An abstract class implementing the Graph edit distance aproximation proposed by Fisher et al.
        The costs for nodes and edges must be defined by inheritance.
    """

    def hec(self, g1, g2):
        c1 = self.edge_deletion(g1)
        c2 = self.edge_insertion(g2)

        c = self.edge_substitution(g1, g2)/2

        for i in range(len(g1)):
            for j in range(len(g2)):
                c1[i] = np.min([c[i, j], c1[i]])
                c2[j] = np.min([c[i, j], c2[j]])

        c = np.sum(c1) + np.sum(c2)
        return c

    def L_graph(self, g1, g2):
        if len(g1)>len(g2):
            l = (len(g1)-len(g2))*np.min(self.node_deletion(g1))
        else:
            l = (len(g2) - len(g1)) * np.min(self.node_insertion(g2))
        return l

    def L_edges(self, u, v):
        if len(u)>len(v):
            l = (len(u)-len(v))*np.min(self.edge_deletion(u))
        else:
            l = (len(v) - len(u)) * np.min(self.edge_insertion(v))
        return l

    """
        Aproximated graph edit distance computation.
    """
    def ged(self, g1, g2):

        # All insertions and substitutions
        assignment1 = np.array([len(g2)+1]*len(g1))
        assignment2 = np.array([len(g1) + 1]*len(g2))

        # Deletion
        d1 = self.node_deletion(g1) + self.edge_deletion(g1.edge.values())/2

        # Insertion
        d2 = self.node_insertion(g2) + self.edge_insertion(g2.edge.values())/2

        # Substitution
        cs = self.node_substitution(g1, g2)

        # edges can be text-named
        g1edges = list(g1.edge.keys())
        g2edges = list(g2.edge.keys())

        for i in range(len(d1)):
            for j in range(len(d2)):
                g1v = g1.edge[g1edges[i]].values()
                g2v = g2.edge[g2edges[j]].values()

                ce = self.hec(g1v, g2v)
                ce = np.max([self.L_edges(g1v, g2v), ce])
                ce = (cs[i,j] + ce/2)/2

                if ce < d1[i]:
                    d1[i] = ce
                    assignment1[i] = j
                if ce < d2[j]:
                    d2[j] = ce
                    assignment2[j] = i

        dist = np.sum(d1) + np.sum(d2)
        dist = np.max([self.L_graph(g1,g2), dist])

        return dist, (assignment1, assignment2)

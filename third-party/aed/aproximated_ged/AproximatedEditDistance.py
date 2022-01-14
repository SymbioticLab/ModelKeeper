#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    AproximatedEditDistance.py

    Riesen, Kaspar, and Horst Bunke. "Approximate graph edit distance computation by means of bipartite graph matching."
    Image and Vision computing 27.7 (2009): 950-959.
"""

from .GraphEditDistance import GraphEditDistance

from scipy.optimize import linear_sum_assignment
import numpy as np

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"


class AproximatedEditDistance(GraphEditDistance):
    """
        An abstract class implementing the Graph edit distance aproximation proposed by Riesen and Bunke.
        The costs for nodes and edges must be defined by inheritance.
    """

    def edge_cost_matrix(self, g1, g2):
        cost_matrix = np.zeros([len(g1)+len(g2),len(g1)+len(g2)])

        #print(f"edge cost matrix {g1.values()}")
        # Insertion
        cost_matrix[len(g1):, 0:len(g2)] = np.inf
        np.fill_diagonal(cost_matrix[len(g1):, 0:len(g2)], self.edge_insertion(g1.values()))

        # Deletion
        cost_matrix[0:len(g1), len(g2):] = np.inf
        np.fill_diagonal(cost_matrix[0:len(g1), len(g2):], self.edge_deletion(g2.values()))

        # Substitution
        cost_matrix[0:len(g1), 0:len(g2)] = self.edge_substitution(g1.values(), g2.values())

        return cost_matrix

    """
        Aproximated graph edit distance for edges. The local structures are matched with this algorithm.
    """
    def edge_ed(self, g1, g2):

        # Compute cost matrix
        cost_matrix = self.edge_cost_matrix(g1, g2)

        # Munkres algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Graph edit distance
        dist = cost_matrix[row_ind, col_ind].sum()

        return dist

    def cost_matrix(self, g1, g2):
        cost_matrix = np.zeros([len(g1)+len(g2),len(g1)+len(g2)])

        #print(f"edge values {g1.edge.values()}")
        # Insertion
        cost_matrix[len(g1):, 0:len(g2)] = np.inf
        np.fill_diagonal(cost_matrix[len(g1):, 0:len(g2)], self.node_insertion(g1)+self.edge_insertion(g1.edge.values()))

        # Deletion
        cost_matrix[0:len(g1), len(g2):] = np.inf
        np.fill_diagonal(cost_matrix[0:len(g1), len(g2):], self.node_deletion(g2)+self.edge_deletion(g2.edge.values()))

        # Substitution
        node_dist = self.node_substitution(g1, g2)

        i1 = 0
        for k1 in g1.nodes():
            i2 = 0
            for k2 in g2.nodes():
                node_dist[i1, i2] += self.edge_ed(g1[k1], g2[k2])
                i2 += 1
            i1 += 1

        cost_matrix[0:len(g1), 0:len(g2)] = node_dist
        return cost_matrix

    """
        Aproximated graph edit distance computation.
    """
    def ged(self, g1, g2):

        # Compute cost matrix
        cost_matrix = self.cost_matrix(g1, g2)

        # Munkres algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Graph edit distance
        dist = cost_matrix[row_ind, col_ind].sum()

        not_assign = np.invert((row_ind >= len(g1)) * (col_ind >= len(g2)))

        return dist, (row_ind[not_assign], col_ind[not_assign])

#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    GraphEditDistance.py

    Abstract class of a generic graph edit distance algorithm.
"""

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"


class GraphEditDistance(object):
    """
        An abstract class representing the Graph edit distance.
    """

    """
        Node edit operations
    """
    def node_substitution(self, g1, g2):
        raise NotImplementedError

    def node_insertion(self, g):
        raise NotImplementedError

    def node_deletion(self, g):
        raise NotImplementedError

    """
        Edge edit operations
    """
    def edge_substitution(self, g1, g2):
        raise NotImplementedError

    def edge_insertion(self, g):
        raise NotImplementedError

    def edge_deletion(self, g):
        raise NotImplementedError

    """
        Graph edit distance computation
    """
    def ged(self, g1, g2):
        raise NotImplementedError

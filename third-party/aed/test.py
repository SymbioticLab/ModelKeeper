#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    main.py

    Distance computation and comparison between different aproximated graph edit distance techniques with the same costs.
"""

#from VanillaAED import VanillaAED
#from VanillaHED import VanillaHED
from aproximated_ged import VanillaAED
from aproximated_ged import VanillaHED

import os
import glob
import itertools

import networkx as nx
import time

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

if __name__ == '__main__':

    path_dataset = './data/'
    name_dataset = 'Letters'

    aed = VanillaAED()
    hed = VanillaHED()

    path_dataset = os.path.join(path_dataset, name_dataset)
    files = glob.glob(path_dataset + '/*.gml')
    for f1, f2 in itertools.combinations_with_replacement(files, 2):
        # Read graphs
        g1 = nx.read_gml(f1)
        g2 = nx.read_gml(f2)
        print(g1.edge)
        assert (1==0)

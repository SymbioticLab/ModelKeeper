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


def read_gml(f1, model_cache):
    if f1 not in model_cache:
        g1 = nx.read_gml(f1)
        model_cache[f1] = g1
    return model_cache[f1]

def get_name(f):
    return f.split('/')[-1].split('.')[0]

if __name__ == '__main__':

    path_dataset = './zoo_gml'
    # name_dataset = './zoo_gml'

    aed = VanillaAED()
    hed = VanillaHED()

    model_cache = {}
    results = {}
    # path_dataset = os.path.join(path_dataset, name_dataset)
    files = [os.path.join(path_dataset, x) for x in os.listdir(path_dataset) if '.gml' in x]

    for f1 in files:
        g1 = read_gml(f1, model_cache)
        results[get_name(f1)] = {}
        for f2 in files:
            g2 = read_gml(f2, model_cache)

            # Distance HED
            hed_start = time.time()
            distHED, _H = aed.ged(g1, g2)
            hed_dur = time.time() - hed_start

            results[get_name(f1)][get_name(f2)] = {'Duration': hed_dur, 'Path': _H, 'GED': distHED}
            print(f"{get_name(f1)} to {get_name(f2)}, {results[get_name(f1)][get_name(f2)]}")
            # print(_A, _H)
            # print(g1.graph['name'] + ' <-> ' + g2.graph['name'] + ' | HED: ' + str(distHED) + ' AED: ' + str(distAED) + ' | ' + str(distHED<=distAED) + (' Exact GED ' if distHED==distAED else ''))
            # print(f"AED: {aed_dur}, HED: {hed_dur}")


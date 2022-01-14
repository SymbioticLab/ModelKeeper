#!/usr/bin/python
# -*- coding: utf-8 -*-

import pytest
import sys, os, glob
import itertools

from aproximated_ged import *


def test_hed():
    path_dataset = './data/'
    name_dataset = 'Letters'

    aed = VanillaAED()

    path_dataset = os.path.join(path_dataset, name_dataset)
    files = glob.glob(path_dataset + '/*.gml')
    for f1, f2 in itertools.combinations_with_replacement(files, 2):
        # Read graphs
        g1 = nx.read_gml(f1)
        g2 = nx.read_gml(f2)

        # Distance
        dist, assignment = aed.ged(g1, g2)

        fig = plot_assignment(g1, g2, assignment)
        fig.savefig('./data/Results/AED/'+g1.graph['class'] + '-' + g2.graph['class'] +'.png')

        print(g1.graph['class'] + ' <-> ' + g2.graph['class'] + ' | Distance: ' + str(dist))

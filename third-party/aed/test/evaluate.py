#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    evaluate.py

    Computes the distance between two folders.
"""

from __future__ import division

from aproximated_ged import VanillaAED
from aproximated_ged import VanillaHED

import glob
import time

from sklearn.metrics import average_precision_score
import networkx as nx
import numpy as np

import argparse

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

# Argument parser
parser = argparse.ArgumentParser(description='Computes the distance all vs all between two folders.')

# Prototypes
parser.add_argument('--folder1', help='Input graphs.', default='./data/Letters/')
parser.add_argument('--folder2', help='Graphs against whose computes the distance.',
                    default='./data/Letters/')
parser.add_argument('--fid', help='Select an specific file from folder1.',
                    default=None)

# Edit Distance
parser.add_argument('--ged', help='Graph edit distance algorithm.', default='vanillaHED')
parser.add_argument('--edit-operations', help='Graph edit distance parameters.', default={})

# Evaluation
parser.add_argument('--retrieval', action='store_true', default=False,
                    help='Computes retrieval instead of classification.')
parser.add_argument('--knn', help='k for k-NN classifier.', default=5)

args = parser.parse_args()


def dist_matrix(fold1, fold2, ged):
    # Find graph files
    files1 = glob.glob(fold1 + '/*.gml')
    files2 = glob.glob(fold2 + '/*.gml')

    if args.fid is not None:
        files1 = [ files1[int(args.fid)-1] ]

    # Distance matrix
    d = np.zeros([len(files1), len(files2)])

    l1 = []
    l2 = []

    for f1 in range(d.shape[0]):
        # Read graph
        g1 = nx.read_gml(files1[f1])
        l1 += g1.graph['class']

        for f2 in range(d.shape[1]):
            # Read graph
            g2 = nx.read_gml(files2[f2])

            if not f1:
                l2 += g2.graph['class']

            # Distance
            dist, _ = ged.ged(g1, g2)

            d[f1, f2] = dist

    return d, l1, l2


def mode_knn(a, axis=1):
    scores = np.unique(np.ravel(a))       # get ALL unique values
    testshape = list(a.shape)

    oldcounts = np.zeros(testshape, dtype=int)
    for score in scores:
        template = (a == score)
        counts = np.sum(template, axis)
        counts = np.repeat(counts[..., None], a.shape[1], axis=1)
        oldcounts[template] = counts[template]
    ind = np.argmax(oldcounts, axis=1)
    return a[np.arange(a.shape[0]),ind]


def classification(d, l1, l2, k=5):
    ind = np.argsort(d, axis=1)
    l2_matrix = np.array([np.array(l2)[ind[i, 0:k]] for i in range(len(l1))])
    pred = mode_knn(l2_matrix)
    acc = np.sum(l1 == pred) / pred.shape[0]
    return acc


def retrievel(d, l1, l2):
    comparison = np.array([[i == j for j in l2] for i in l1])
    mAP = average_precision_score(comparison, np.exp(-d))
    return mAP


def evaluation(folder1, folder2, ged, criterium):
    startC = time.time()
    # Distance matrix computation
    d, l1, l2 = dist_matrix(folder1, folder2, ged)

    startE = time.time()

    # Evaluation
    performance = criterium(d, l1, l2)

    print('Performance: {}\t'
          'Time Computation {}\t'
          'Time Evaluation {}'
          .format(performance, startE-startC, time.time()-startE))


if __name__ == '__main__':
    ged = {
        'vanillaaed': VanillaAED(**args.edit_operations),
        'vanillahed': VanillaHED(**args.edit_operations)
    }.get(args.ged.lower(), None)

    if args.retrieval:
        print('Retrieval')
        criterium = retrievel
    else:
        print('Classification')
        criterium = lambda d, l1, l2: classification(d, l1, l2, args.knn)

    if ged is not None:
        evaluation(args.folder1, args.folder2, ged, criterium)
    else:
        raise NameError('GED argument not implemented')

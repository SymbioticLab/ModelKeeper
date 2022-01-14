#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Plotter.py

    Set of tools to visualize graphs. Assumes label 'coord' as the 2D position of the nodes.
"""

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def plot_graph(g, show=False, save_path=''):
    fig = plt.figure()
    position = {k: v['coord'] for k, v in g.node.items()}

    center = np.mean(position.values(),axis=0)
    max_pos = np.max(np.abs(position.values()-center))

    nx.draw(g, pos=position)

    plt.ylim([center[1]-max_pos-0.5, center[1]+max_pos+0.5])
    plt.xlim([center[0]-max_pos-0.5, center[0]+max_pos+0.5])

    if show:
        plt.show()
    fig.savefig(save_path)


def plot_assignment(g1, g2, assignment, show=False):
    g_comp = nx.disjoint_union(g1, g2)

    position = {k: v['coord'] for k, v in g_comp.node.items()}

    center1 = np.mean(list(position.values())[0:len(g1)],axis=0)
    max_pos1 = np.max(np.abs(list(position.values())[0:len(g1)]-center1))

    center2 = np.mean(list(position.values())[len(g1):], axis=0)
    max_pos2 = np.max(np.abs(list(position.values())[len(g1):] - center2))

    keys_modify = list(position.keys())[len(g1):]
    for x in keys_modify:
        position[x][0] += 2 * (np.max([max_pos1, max_pos2]) + 0.5)

    ag1, ag2 = assignment

    edgelist =  []
    nodelist_ins = []
    nodelist_del = []
    for i1, i2 in zip(ag1, ag2):
        if i1<len(g1):
            if i2<len(g2):
                # Substitution
                g_comp.add_edge(i1, i2+len(g1))
                edgelist += [(i1, i2+len(g1))]
            else:
                # Deletion
                nodelist_del += [i1]
        else:
            # Insertion
            nodelist_ins += [i2+len(g1)]

    center = np.mean(list(position.values()), axis=0)
    max_pos = np.max(np.abs(list(position.values()) - center))

    fig = plt.figure()

    nx.draw_networkx_nodes(g_comp, position,
                           nodelist=[item for item in g_comp.nodes() if item not in nodelist_ins + nodelist_del],
                           node_color='black',
                           node_size=200)

    nx.draw_networkx_edges(g_comp, position,
                           edgelist=[item for item in g_comp.edges() if item not in edgelist]
                           )

    nx.draw_networkx_edges(g_comp, position,
                           edgelist=edgelist,
                           width=3, alpha=0.5, edge_color='b',
                           style='dashed')

    nx.draw_networkx_nodes(g_comp, position,
                           nodelist=nodelist_ins,
                           node_color='g',
                           node_size=500,
                           alpha=0.8)

    nx.draw_networkx_nodes(g_comp, position,
                           nodelist=nodelist_del,
                           node_color='r',
                           node_size=500,
                           alpha=0.8)

    plt.ylim([center[1] - max_pos - 0.5, center[1] + max_pos + 0.5])
    plt.xlim([center[0] - max_pos - 0.5, center[0] + max_pos + 0.5])
    plt.axis('off')
    if show:
        plt.show()

    return fig


def plot_assignment_hausdorff(g1, g2, assignment, show=False):
    g_comp = nx.disjoint_union(g1, g2)
    g_comp = g_comp.to_directed()

    position = {k: v['coord'] for k, v in g_comp.node.items()}

    center1 = np.mean(list(position.values())[0:len(g1)],axis=0)
    max_pos1 = np.max(np.abs(list(position.values())[0:len(g1)]-center1))

    center2 = np.mean(list(position.values())[len(g1):], axis=0)
    max_pos2 = np.max(np.abs(list(position.values())[len(g1):] - center2))

    keys_modify = list(position.keys())[len(g1):]
    for x in keys_modify:
        position[x][0] += 2 * (np.max([max_pos1, max_pos2]) + 0.5)

    ag1, ag2 = assignment

    edgelist =  []
    nodelist_ins = []
    nodelist_del = []

    for i1, i2 in zip(range(len(ag1)), ag1):
        if i2<len(g2):
            # Substitution
            g_comp.add_edge(i1, i2+len(g1))
            edgelist += [(i1, i2+len(g1))]
        else:
            # Deletion
            nodelist_del += [i1]

    for i1, i2 in zip(ag2, range(len(ag2))):
        if i1<len(g1):
            # Substitution
            g_comp.add_edge(i2+len(g1), i1)
            edgelist += [(i2+len(g1), i1)]
        else:
            # Insertion
            nodelist_ins += [i2+len(g1)]

    center = np.mean(position.values(), axis=0)
    max_pos = np.max(np.abs(position.values() - center))

    fig = plt.figure()

    nx.draw_networkx_nodes(g_comp, position,
                       nodelist=[item for item in g_comp.nodes() if item not in nodelist_ins + nodelist_del],
                       node_color='black',
                       node_size=200)


    nx.draw_networkx_edges(g_comp, position,
                           edgelist=[item for item in g_comp.edges() if item not in edgelist],
                           arrows=False)

    nx.draw_networkx_edges(g_comp, position,
                           edgelist=edgelist,
                           width=3, alpha=0.5, edge_color='b',
                           style='dashed')

    nx.draw_networkx_nodes(g_comp, position,
                       nodelist=nodelist_ins,
                       node_color='g',
                       node_size=500,
                       alpha=0.8)

    nx.draw_networkx_nodes(g_comp, position,
                       nodelist=nodelist_del,
                       node_color='r',
                       node_size=500,
                       alpha=0.8)

    plt.ylim([center[1] - max_pos - 0.5, center[1] + max_pos + 0.5])
    plt.xlim([center[0] - max_pos - 0.5, center[0] + max_pos + 0.5])
    plt.axis('off')
    if show:
        plt.show()

    return fig
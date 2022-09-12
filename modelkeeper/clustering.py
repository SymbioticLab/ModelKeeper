#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, with_statement

import logging
import multiprocessing
import random
from operator import __eq__, itemgetter

_MAX_ITER = int(1e3)
random.seed(1)


class Medoid(object):
    __slots__ = ['kernel', 'elements']

    def __init__(self, kernel, elements=None):
        self.kernel = kernel
        self.elements = [] if elements is None else elements

    def __repr__(self):
        return 'Medoid({0}, {1})'.format(self.kernel, self.elements)

    def __iter__(self):
        return iter(self.elements)

    def compute_kernel(self, distance):
        return min(self, key=lambda e: sum(distance(e, other)
                                           for other in self))

    def compute_diameter(self, distance):
        return max(distance(a, b) for a in self for b in self)


def _k_medoids_spawn_once(points, k, distance,
                          equality=__eq__,
                          max_iterations=_MAX_ITER,
                          verbose=True):
    """K-medoids algorithm with one spawn of medoid kernels.

    :param points:    the list of points
    :param k:         the number of clusters
    :param distance:  the distance function, distance(p, q) = ||q - p||
    :param max_iterations: the maximum number of iterations
    :param verbose:   verbosity
    :returns:         the partition, structured as \
        a list of [kernel of the cluster, [elements in the cluster]]

    >>> points = [1, 2, 3, 4, 5, 6, 7]
    >>> def distance(a, b):
    ...     return abs(b - a)
    >>> diameter, medoids = _k_medoids_spawn_once(points, k=2, distance=distance) #doctest: +SKIP
    * New chosen kernels: [6, 3]
    * Iteration over after 3 steps, max diameter 3
    """
    if k <= 0:
        raise ValueError('Number of medoids must be strictly positive')
    if k > len(points):
        raise ValueError('Number of medoids exceeds number of points')

    # Medoids initialization
    medoids = [Medoid(kernel=p) for p in random.sample(list(points), k)]
    # if verbose:
    #    print('* New chosen kernels: {0}'.format([m.kernel for m in medoids]))

    for n in range(1, 1 + max_iterations):
        # Resetting medoids
        for m in medoids:
            m.elements = []

        # Putting points in closest medoids
        for p in points:
            closest_medoid = min(medoids, key=lambda m: distance(m.kernel, p))
            closest_medoid.elements.append(p)

        # Removing empty medoids
        medoids = [m for m in medoids if m.elements]

        # Electing new kernels for each medoids
        change = False
        for m in medoids:
            new_kernel = m.compute_kernel(distance)
            if not equality(new_kernel, m.kernel):
                m.kernel = new_kernel
                change = True

        if not change:
            break

    diameter = max(m.compute_diameter(distance) for m in medoids)
    # if verbose:
    #    print('* Iteration over after {} steps, max diameter {}'.format(n, diameter))

    return diameter, medoids


def k_medoids(points, k, distance, spawn,
              equality=__eq__,
              max_iterations=_MAX_ITER,
              verbose=True,
              threads=1):
    """
    Same as _k_medoids_spawn_once, but we iterate also the spawning process.
    We keep the minimum of the biggest diameter as a reference for the best spawn.

    :param points:    the list of points
    :param k:         the number of clusters
    :param distance:  the distance function, distance(p, q) = ||q - p||
    :param spawn:     the number of spawns
    :param max_iterations: the maximum number of iterations
    :param verbose:   boolean, verbosity status
    :returns:         the partition, structured as \
        a list of [kernel of the cluster, [elements in the cluster]]
    """

    # Here the result of _k_medoids_spawn_once function is a tuple containing
    # in the second element the diameter of the biggest medoid, so the min
    # function will return the best medoids arrangement, in the sense that the
    # diameter max will be minimum
    if len(points) == 0:
        logging.error(f"Clustering size can not be zero")
        return float('inf'), []

    pool = multiprocessing.Pool(processes=threads)

    ans = [
        pool.apply_async(
            _k_medoids_spawn_once,
            (points,
             k,
             distance,
             equality,
             max_iterations,
             verbose)) for _ in range(spawn)]

    pool.close()
    pool.join()

    best_diameter, best_medoids = min(
        [(x.get()) for x in ans], key=lambda k: k[0])

    return best_diameter, best_medoids


def k_medoids_auto_k(points, distance, spawn, diam_max,
                     equality=__eq__,
                     max_iterations=_MAX_ITER,
                     start_k=1,
                     threads=1,
                     verbose=True):
    """
    Same as k_medoids, but we increase the number of clusters until we have a
    good enough similarity between points.

    :param points:    the list of points
    :param diam_max:  the maximum diameter allowed, otherwise \
        the algorithm will start over and increment the number of clusters
    :param distance:  the distance function, distance(p, q) = ||q - p||
    :param spawn:     the number of spawns
    :param iteration: the maximum number of iterations
    :param verbose:   verbosity
    :returns:         the partition, structured as \
        a list of [kernel of the cluster, [elements in the cluster]]
    """
    if len(points) == 0:
        # we do not test `if not points` to keep things compatible with numpy
        # arrays
        raise ValueError('No points given!')

    kw = {
        'distance': distance,
        'equality': equality,
        'spawn': spawn,
        'max_iterations': max_iterations,
        'verbose': verbose,
        'threads': threads,
    }

    # Binary search: diameter decreases as # of clusters increases
    left, right = start_k, len(points)
    mid = 0

    while left < right:
        mid = left + (right - left) // 2

        diameter, medoids = k_medoids(points, mid, **kw)

        if diameter > diam_max:
            print(
                '*** {} clusters, diameter too big {:.3f} > {:.3f}\n'.format(mid, diameter, diam_max))
            left = mid + 1
        else:
            print(
                '*** {} clusters, diameter too small {:.3f} < {:.3f}\n'.format(mid, diameter, diam_max))
            right = mid

    diameter, medoids = k_medoids(points, right, **kw)

    if verbose:
        print(
            '\n\n*** Diameter ok {:.3f} <= {:.3f}'.format(diameter, diam_max))
        print('*** Stopping, {} clusters enough ({} points initially)'.format(right, len(points)))

    return diameter, medoids

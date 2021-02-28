#!/usr/bin/env python
from __future__ import print_function
import numpy

class AlignmentOperator(object):
    __matchscore = 1
    __mismatchscore = -1
    __gap = -2

    def __init__(self, parent, child, globalAlign=True,
                 matchscore=__matchscore, mismatchscore=__mismatchscore,
                 gapscore=__gap):
        self._mismatchscore = mismatchscore
        self._matchscore = matchscore
        self._gap = gapscore
        self.child    = child
        self.parent       = parent
        self.matchidxs  = None
        self.parentidxs    = None
        self.globalAlign = globalAlign

        self.parentidx_order = self.parent.topological_sorting()
        matches = self.alignChildToParent()
        self.matchidxs, self.parentidxs = matches

    def alignmentStrings(self):
        return ("".join([self.child.vs[i]['name'] if i is not None else "-" for i in self.matchidxs]),
                "".join([self.parent.vs[j]['name'] if j is not None else "-" for j in self.parentidxs]))

    def matchscore(self, c1, c2):
        if c1 == c2:
            return self._matchscore
        else:
            return self._mismatchscore

    def alignChildToParent(self):
        """Align node to parent, following same approach as smith waterman
        example"""
        nodeIDtoIndex, nodeIndexToID, scores, backStrIdx, backGrphIdx = self.initializeDynamicProgrammingData(self.parentidx_order)

        # Dynamic Programming
        for i, pidx in enumerate(self.parentidx_order):
            node = self.parent.vs[pidx]
            pbase = node['attr']

            for j, cnode in enumerate(self.child.vs):
                sbase = cnode['attr']
                # add all candidates to a list, pick the best
                candidates = [(scores[i+1, j] + self._gap, i+1, j, "INS")]
                for predIndex in self.prevIndices(node, nodeIDtoIndex):
                    candidates += [(scores[predIndex+1, j+1] + self._gap, predIndex+1, j+1, "DEL")]
                    candidates += [(scores[predIndex+1, j] + self.matchscore(sbase, pbase), predIndex+1, j, "MATCH")]

                scores[i+1, j+1], backGrphIdx[i+1, j+1], backStrIdx[i+1, j+1], movetype = max(candidates)

                if not self.globalAlign and scores[i+1, j+1] < 0:
                    scores[i+1, j+1] = 0.
                    backGrphIdx[i+1, j+1] = -1
                    backStrIdx[i+1, j+1] = -1

        return self.backtrack(scores, backStrIdx, backGrphIdx, nodeIndexToID, self.parentidx_order)

    def prevIndices(self, node, nodeIDtoIndex):
        """Return a list of the previous dynamic programming table indices
           corresponding to predecessors of the current node."""
        prev = []
        for edge in node.in_edges():
            prev.append(nodeIDtoIndex[edge.source])

        # if no predecessors, point to just before the parent
        if len(prev) == 0:
            prev = [-1]
        return prev

    def initializeDynamicProgrammingData(self, ni):
        """Initalize the dynamic programming tables:
            @ni: re-index graph nodes
            - set up scores array
            - set up backtracking array
            - create index to Node ID table and vice versa
        """
        l1 = len(self.parent.vs)
        l2 = len(self.child.vs)

        nodeIDtoIndex = {}
        nodeIndexToID = {-1: None}
        # generate a dict of (nodeID) -> (index into nodelist (and thus matrix))
        for (index, nidx) in enumerate(ni):
            node = self.parent.vs[nidx]
            nodeIDtoIndex[node.index] = index
            nodeIndexToID[index] = node.index

        # Dynamic Programming data structures; scores matrix and backtracking
        # matrix
        scores = numpy.zeros((l1+1, l2+1), dtype=numpy.int)

        # initialize insertion score
        # if global align, penalty for starting at head != 0
        if self.globalAlign:
            scores[0, :] = numpy.arange(l2+1)*self._gap

            for (index, nidx) in enumerate(ni):
                node = self.parent.vs[nidx]
                prevIdxs = self.prevIndices(node, nodeIDtoIndex)
                best = scores[prevIdxs[0]+1, 0]
                for prevIdx in prevIdxs:
                    best = max(best, scores[prevIdx+1, 0])
                scores[index+1, 0] = best + self._gap

        # backtracking matrices
        backStrIdx = numpy.zeros((l1+1, l2+1), dtype=numpy.int)
        backGrphIdx = numpy.zeros((l1+1, l2+1), dtype=numpy.int)

        return nodeIDtoIndex, nodeIndexToID, scores, backStrIdx, backGrphIdx

    def backtrack(self, scores, backStrIdx, backGrphIdx, nodeIndexToID, ni):
        """Backtrack through the scores and backtrack arrays.
           Return a list of child indices and node IDs (not indices, which
           depend on ordering)."""
        besti, bestj = scores.shape
        besti -= 1
        bestj -= 1
        if not self.globalAlign:
            besti, bestj = numpy.argwhere(scores == numpy.amax(scores))[-1]
        else:
            # still have to find best final index to start from
            terminalIndices = [index for (index, pidx) in enumerate(ni) if self.graph.vs[pidx].outdegree==0]

            besti = terminalIndices[0] + 1
            bestscore = scores[besti, bestj]
            for i in terminalIndices[1:]:
                score = scores[i+1, bestj]
                if score > bestscore:
                    bestscore, besti = score, i+1

        matches = []
        strindexes = []
        while (self.globalAlign or scores[besti, bestj] > 0) and not(besti == 0 and bestj == 0):
            nexti, nextj = backGrphIdx[besti, bestj], backStrIdx[besti, bestj]
            curstridx, curnodeidx = bestj-1, nodeIndexToID[besti-1]

            strindexes.insert(0, curstridx if nextj != bestj else None)
            matches.insert(0, curnodeidx if nexti != besti else None)

            besti, bestj = nexti, nextj

        return strindexes, matches


# cython: language_level=3

__version__ = "1.0.1"

from libcpp.vector cimport vector
from libcpp cimport bool as cbool
from libc.math cimport isnan

cdef extern from "chu_liu_edmonds_internal.h":
    cdef void c_chu_liu_edmonds(
            vector[cbool] *disabled,
            vector[vector[int]] *candidate_heads,
            vector[vector[double]] *candidate_scores,
            vector[int] *heads,
            double *value);

def chu_liu_edmonds(double[:,:] score_matrix):
    """

    :param score_matrix: an N by N matrix where the i,j-th cell is the score
        of i having j as a head. Index 0 is the artificial root node.
    :param tol: Ignore scores that are closer than tol to zero.
    :return:
    """
    # The size of the sentence includes the root at index 0
    cdef size_t sentence_len = len(score_matrix)
    cdef vector[vector[int]] candidate_heads
    cdef vector[vector[double]] candidate_scores
    cdef vector[int] heads = vector[int](sentence_len, -1)
    cdef vector[cbool] disabled = vector[cbool](sentence_len, <cbool> False)
    cdef double tree_score = 0

    candidate_scores.resize(sentence_len)
    candidate_heads.resize(sentence_len)

    assert score_matrix.shape[0] == score_matrix.shape[1], "Score matrix must be square"

    cdef int dep_i, head_i
    cdef double edge_score
    for dep_i in range(1, score_matrix.shape[0]):
        for head_i in range(score_matrix.shape[1]):
            edge_score = score_matrix[dep_i, head_i]
            if not isnan(edge_score):
                candidate_heads[dep_i].push_back(head_i)
                candidate_scores[dep_i].push_back(edge_score)


    c_chu_liu_edmonds(disabled=&disabled, candidate_heads=&candidate_heads, candidate_scores=&candidate_scores,
                    heads=&heads, value=&tree_score)

    # Convert heads format
    return heads, tree_score

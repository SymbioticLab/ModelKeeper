# Chu-Liu-Edmonds Algorithm from TurpoParser.

This package wraps the Chu-Liu-Edmonds maximum spanning algorithm from
TurboParser for use within Python.

The original package was made by https://github.com/andersjo/dependency_decoding .

## Documentation

The package provides a function `chu_liu_edmonds` which accepts a _N×N_ score
matrix as argument, where _N_ is the sentence length, including the artificial
root node. The _(i,j)_-th cell is the score for the edge _j→i_.
In other words, a row gives the scores for the different heads of a dependent.

A `np.nan` cell value informs the algorithm to skip the edge.

Example usage:
```python
import numpy as np
from ufal.chu_liu_edmonds import chu_liu_edmonds

np.random.seed(42)
score_matrix = np.random.rand(3, 4)
heads, tree_score = chu_liu_edmonds(score_matrix)
print(heads, tree_score)
```

## Install

Binary wheels of the package are provided, just run
```
pip install ufal.chu_liu_edmonds
```

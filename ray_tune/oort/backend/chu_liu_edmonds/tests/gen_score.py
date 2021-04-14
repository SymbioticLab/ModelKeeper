
import os, pickle
import numpy as np 

def process_file(file):
    ans = np.full((141, 141), float('-inf'))
    with open(file) as fin:
        lines = fin.readlines()

        file_set = {}
        for line in lines:
            if 'score' not in line:
                continue
            items = line.split('/')
            src = items[8].split(',')[0]
            dst = items[-1].split(')')[0]

            if src not in file_set:
                file_set[src] = len(file_set)
            if dst not in file_set:
                file_set[dst] = len(file_set)

            score = float(items[-1].split()[-1])
            ans[file_set[src], file_set[dst]] = score
            #ans.append((file_set[src], file_set[dst], score))

    with open('score_graph.pkl', 'wb') as fout:
        pickle.dump(ans, fout)

process_file('score.in')

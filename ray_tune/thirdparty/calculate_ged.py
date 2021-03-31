import argparse
import contextlib
import itertools
import json
import random
import time
from pathlib import Path

import joblib
import networkx as nx
from interruptingcow import Quota, timeout
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from utils import make_graph


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback:
        def __init__(self, time, index, parallel):
            self.index = index
            self.parallel = parallel

        def __call__(self, index):
            tqdm_object.update()
            if self.parallel._original_iterator is not None:
                self.parallel.dispatch_next()

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def calc_ged(recepie1, recepie2, timeout_val=600):
    start_time = time.time()
    G1 = make_graph(recepie1)
    G2 = make_graph(recepie2)
    ged = None

    try:
        status = "OK"
        with timeout(Quota(timeout_val), exception=RuntimeError):
            for ged in nx.optimize_graph_edit_distance(G1, G2, lambda n1, n2: n1['op'] == n2['op']):
                pass

    except RuntimeError as e:
        status = "Timeout"

    except Exception as e:
        status = "Exception: " + str(e)

    return {
        "recepie_i": recepie1,
        "recepie_j": recepie2,
        "ged": ged,
        "time": time.time() - start_time,
        "status": status
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate GED')
    parser.add_argument('--recepies', type=str, default="./new_recepies_fix.json",
                        help='path to JSON file with recepies')
    parser.add_argument('--num', type=int, default=10,
                        help='number of random recepies for calculating GED to all another')
    parser.add_argument('--timeout', type=int, default=600, help="timeout for calculating one GED value in seconds")
    parser.add_argument('--n_jobs', type=int, default=-2,
                        help="n_jobs in skit learn style")
    parser.add_argument('--num_parts', type=int, default=10,
                        help="Num results parts for saving")

    args = parser.parse_args()
    
    with open(args.recepies, "r") as f:
        recepies = json.load(f)

    key_recepies = random.sample(recepies, args.num)
    part_size = len(recepies)//args.num_parts
    for part in range(1, args.num_parts+1):
        _recepies = recepies[(part-1)*part_size:part*part_size]
        combs = list(itertools.product(key_recepies, _recepies))

        with tqdm_joblib(tqdm(desc="GED part {} of {}".format(part, args.num_parts), total=len(combs))) as progress_bar:
            results = Parallel(n_jobs=args.n_jobs, backend='multiprocessing')(delayed(calc_ged)(r1, r2, args.timeout) for r1, r2 in combs)

        with open("GED_CALC_RESULTS_part_{}.json".format(part), 'w') as f:
            json.dump(results, f)

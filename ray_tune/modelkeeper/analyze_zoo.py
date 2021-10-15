import onnx
import numpy
import networkx as nx
import time, sys, os
import functools, collections
from matchingopt import Oort
import logging
from onnx import numpy_helper
import multiprocessing
import torch
import heapq
from multiprocessing import Manager
import ctypes
import json, gc

# Call C backend
clib_matcher = ctypes.cdll.LoadLibrary('/users/fanlai/ModelKeeper/ray_tune/oort/backend/bin/matcher.so')
clib_matcher.get_matching_score.restype = ctypes.c_char_p

sys.setrecursionlimit(10000)
logging.basicConfig(filename='logging', level=logging.INFO)


def get_mapped(file):
    black_list = set()
    with open(file) as fin:
        lines = fin.readlines()
        for line in lines:
            if 'Find best mappings' in line:
                model_name = line.split('/')[3].split()[0]
                black_list.add(model_name)
    return black_list

def analyze_zoo():
    import argparse

    start_time = time.time()
    zoo_path = '/mnt/weight/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--zoo_path', type=str, default=zoo_path)
    parser.add_argument('--num_of_processes', type=int, default=20)

    args = parser.parse_args()
    mapper = Oort(args)

    models = sorted(os.listdir(zoo_path))

    #black_list = get_mapped('/users/fanlai/torchcv_scores')
    #models = [x for x in os.listdir(zoo_path) if x not in black_list]
    #print(models)
    #print(len(models))
    for idx, model_name in enumerate(models):
        child_onnx_path = os.path.join(zoo_path, model_name)
        child, child_onnx = mapper.load_model_meta(child_onnx_path)
        child.graph['model_id'] = str(idx)

        # find the best mapping from the zoo
        parent, mappings, best_score = mapper.get_best_mapping(child, set([]), model_name, return_weight=False)

        gc.collect()

    print("==============")
    print(f"total duration is {(time.time()-start_time)/1000.0} sec")


def analyze_zoo_folder():
    import argparse

    start_time = time.time()
    zoo_path = '/mnt/zoo/transformers/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--zoo_path', type=str, default=zoo_path)
    parser.add_argument('--num_of_processes', type=int, default=40)

    args = parser.parse_args()
    mapper = Oort(args)

    model_folders = os.listdir(zoo_path)
    models = []
    for idx, model_path in enumerate(model_folders):
        model_name = [x for x in os.listdir(os.path.join(zoo_path, model_path)) if '.onnx' in x]
        if len(model_name) == 1:
            models.append(os.path.join(zoo_path, model_path, model_name[0]))
            mapper.add_to_zoo(models[-1], idx)
            print(f"===Add {models[-1]} to zoo...")

    # models = os.listdir(zoo_path)
    for idx, model_name in enumerate(models):
        child_onnx_path = model_name #os.path.join(zoo_path, model_name)
        child, child_onnx = mapper.load_model_meta(child_onnx_path)
        child.graph['model_id'] = str(idx)

        # find the best mapping from the zoo
        parent, mappings, best_score = mapper.get_best_mapping(child, set([]), model_name.split('/')[-1], return_weight=False)

        gc.collect()

    print("==============")
    print(f"total duration is {(time.time()-start_time)/1000.0} sec")

analyze_zoo_folder()

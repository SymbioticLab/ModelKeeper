import collections
import ctypes
import functools
import gc
import heapq
import json
import logging
import multiprocessing
import os
import sys
import time
from multiprocessing import Manager

import networkx as nx
import numpy
import onnx
import torch
from matcher import ModelKeeper
from onnx import numpy_helper

# Call C backend
clib_matcher = ctypes.cdll.LoadLibrary('./backend/bin/matcher.so')
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
    from config import modelkeeper_config

    start_time = time.time()
    zoo_path = "/users/fanlai/experiment/keeper/model_zoo"  # '/mnt/zoo'

    modelkeeper_config.zoo_path = zoo_path
    mapper = ModelKeeper(modelkeeper_config)

    # ["/users/fanlai/model_zoo/ShuffleNetV2_net_size_2_@0.7142.onnx"]
    models = [x for x in os.listdir(zoo_path) if '.onnx' in x]

    #black_list = get_mapped('/users/fanlai/torchcv_scores')
    #models = [x for x in os.listdir(zoo_path) if x not in black_list]
    # print(models)
    # print(len(models))
    all_models = [os.path.join(zoo_path, x) for x in models]
    for idx, model_name in enumerate(models):
        try:
            child_onnx_path = os.path.join(zoo_path, model_name)
            # child, child_onnx = mapper.load_model_meta(child_onnx_path)
            # child.graph['model_id'] = str(idx)

            black_list = set(all_models[:])
            black_list.discard(child_onnx_path)
            black_list = set()
            # find the best mapping from the zoo
            weights, meta_data = mapper.map_for_onnx(
                child_onnx_path, black_list, model_name)
            print(meta_data)
            gc.collect()
        except Exception as e:
            print(e)

    print("==============")
    print(f"total duration is {(time.time()-start_time)/1000.0} sec")


def analyze_zoo_folder():
    from config import modelkeeper_config

    start_time = time.time()
    zoo_path = '/users/fanlai/experiment/data/my_zoo'

    modelkeeper_config.zoo_path = zoo_path
    mapper = ModelKeeper(modelkeeper_config)

    model_folders = [
        x for x in os.listdir(zoo_path) if os.path.isdir(
            os.path.join(
                zoo_path, x))]
    models = []  # model_folders#["/users/fanlai/experiment/data/my_zoo/funnel_transformer_small/funnel_transformer_small.onnx"]
    #model_folders = models
    for idx, model_path in enumerate(model_folders):
        model_name = [
            x for x in os.listdir(
                os.path.join(
                    zoo_path,
                    model_path)) if '.onnx' in x]
        if len(model_name) == 1:
            models.append(os.path.join(zoo_path, model_path, model_name[0]))
            mapper.add_to_zoo(models[-1])
            print(f"===Add {models[-1]} to zoo...")

    # models = os.listdir(zoo_path)
    for idx, model_name in enumerate(models):
        child_onnx_path = model_name  # os.path.join(zoo_path, model_name)
        # child, child_onnx = mapper.load_model_meta(child_onnx_path)
        # child.graph['model_id'] = str(idx)
        black_list = set(models[:])
        black_list.discard(child_onnx_path)
        # find the best mapping from the zoo
        weights, meta_data = mapper.map_for_onnx(
            child_onnx_path, black_list, model_name.split('/')[-1])
        print(meta_data)
        gc.collect()

    print("==============")
    print(f"total duration is {(time.time()-start_time)/1000.0} sec")


# analyze_zoo_folder()
analyze_zoo()

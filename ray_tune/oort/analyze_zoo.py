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


def analyze_zoo():
	import argparse

	start_time = time.time()
	zoo_path = '/mnt/weight/'

	parser = argparse.ArgumentParser()
	parser.add_argument('--zoo_path', type=str, default=zoo_path)	
	parser.add_argument('--num_of_processes', type=int, default=64)

	args = parser.parse_args()
	mapper = Oort(args)

	models = os.listdir(zoo_path)
	for idx, model_name in enumerate(models):
		child_onnx_path = os.path.join(zoo_path, model_name)
		child, child_onnx = mapper.load_model_meta(child_onnx_path)
		child.graph['model_id'] = str(idx)

		# find the best mapping from the zoo
		parent, mappings, best_score = mapper.get_best_mapping(child, set([]), model_name)

		gc.collect()

	print("==============")
	print(f"total duration is {(time.time()-start_time)/1000.0} sec")

analyze_zoo()



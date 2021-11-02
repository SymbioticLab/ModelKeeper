'''
    Offline register of model keeper client service
'''

from modelkeeper.config import modelkeeper_config
from modelkeeper.clientservice import ModelKeeperClient

import argparse
import logging
import pickle
import torch
import time
import os

log_path = './modelkeeper_log'
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                datefmt='%H:%M:%S',
                level=logging.INFO,
                handlers=[
                    logging.FileHandler(log_path, mode='a'),
                    logging.StreamHandler()
                ])

parser = argparse.ArgumentParser(description="ModelKeeper offline client APIs")
parser.add_argument('--task', type=str, default='cv')
parser.add_argument('--model_file', type=str, default=None)
parser.add_argument('--export_path', type=str, default=None)
parser.add_argument('--accuracy', type=float, default=-1)

def register_model(model_file, export_path, accuracy):
    with open(model_file, 'rb') as fin:
        model = pickle.load(fin)
        dummpy_input = pickle.load(fin)

    #os.remove(model_file)
    torch.onnx.export(model, dummpy_input, export_path, export_params=True, verbose=0, training=1)

    # register model to the zoo
    modelkeeper_client = ModelKeeperClient(modelkeeper_config)
    modelkeeper_client.register_model_to_zoo(export_path, accuracy=accuracy)
    modelkeeper_client.stop()
    os.remove(export_path)


args, unknown = parser.parse_known_args()

logging.info(f"Start to upload {args.model_file}")
register_model(args.model_file, args.model_file, args.accuracy)
logging.info(f"Successfully upload model {args.model_file} to the zoo")

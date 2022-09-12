import os
import pickle
from random import Random

import torch
from models import get_cell_based_tiny_net
from torch.autograd import Variable


def GenerateConfig(n, path):
    """
    n : number of models
    path : meta file path
    """

    fr = open(path,'rb')
    config_list = pickle.load(fr)

    rng = Random()
    rng.seed(0)
    rng.shuffle(config_list)

    return config_list[:n]

def export_onnx():
  dummy_input = torch.rand(8, 3, 32, 32) #  batch:32; 3 channels; 32 x 32 size
  path = '/gpfs/gpfs0/groups/chowdhury/fanlai/model_zoo/imagenet120/motivation'
  cnt = 0
  params_set = set()

  conf_list = GenerateConfig(500, "/gpfs/gpfs0/groups/chowdhury/dywsjtu/ImageNet16-120_config.pkl")

  for idx, conf in enumerate(conf_list):
    model = get_cell_based_tiny_net(conf)
    torch.onnx.export(model, dummy_input, os.path.join(path, f"model_{idx}.onnx"), 
                        export_params=False, verbose=0, training=1)
    print(f"Successfully generate {idx}")
    cnt += 1

    except Exception as e:
      print(f"{idx} failed due to {e}")

  print("============")
  print(f"Generate {cnt} models in total, failed {len(conf_list)-cnt} models")


export_onnx()


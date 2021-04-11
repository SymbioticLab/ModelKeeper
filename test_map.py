# Oort lib
from oort.config import args
from oort.matchingopt import Oort
import torchvision.models as models
import torch


import time, os

def simulator(path):
    mapper = Oort(args)
    num_of_matched = 0

    for i in range(100):
        model_path = os.path.join(path, f"model_{i}.onnx")
        child, _ = mapper.load_model_meta(model_path)
        parent, mappings, best_score = mapper.get_best_mapping(child)
        if parent is not None:
            weights, num_of_matched = mapper.warm_weights(parent, child, mappings)
        mapper.add_to_zoo(model_path)
        print("Child model {}".format(i))

def main():
    model_type = 'resnet50'
    model = models.__dict__[model_type](num_classes=10)

    start_time = time.time()
    mapper = Oort(args)

    print("Initiate matching operator takes {:.2f} sec".format(time.time()-start_time))
    
    mapper.map_for_model(model, torch.rand(8, 3, 32, 32))

#main()
simulator('/gpfs/gpfs0/groups/chowdhury/fanlai/model_zoo/nasbench201')


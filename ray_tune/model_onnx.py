import torchvision.models as models
import torch
import os, pickle, sys
from models import get_cell_based_tiny_net
import models.cell_infers

sys.modules['models'] = models
sys.modules['models.cell_infers'] = models.cell_infers

def remove_dummy_model(path):
    files = os.listdir(path)

    for file in files:
        model_id = int(file.split('.')[0].split('_')[-1])
        if model_id > 102:
            os.system(f'rm {os.path.join(path, file)}')

def gen_model_zoo(path):
    dummy_input = torch.rand(8, 3, 32, 32) #  batch:32; 3 channels; 32 x 32 size
    for i in range(101):
        model_name = f'model_{i}.pth'
        with open(os.path.join(path, model_name), 'rb') as fin:
            model = pickle.load(fin)

        torch.onnx.export(model, dummy_input, os.path.join(path, f"model_{i}.onnx"), 
                                    export_params=True, verbose=0, training=1)

        print(f"Done {i} ...")
#remove_dummy_model('/gpfs/gpfs0/groups/chowdhury/fanlai/model_zoo/nasbench201/')
gen_model_zoo('/gpfs/gpfs0/groups/chowdhury/fanlai/model_zoo/nasbench201/')


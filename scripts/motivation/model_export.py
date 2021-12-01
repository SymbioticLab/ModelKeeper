import torchvision.models as models
import torch
import os, pickle, sys

def gen_model_zoo(path):
    dummy_input = torch.rand(8, 3, 32, 32) #  batch:32; 3 channels; 32 x 32 size
    files = [os.path.join(path, x) for x in os.listdir(path) if x.endswith('.pkl')]

    for file in files:
        with open(file, 'rb') as fin:
            _, _, model = pickle.load(fin), pickle.load(fin), pickle.load(fin)

        model = model.to(device='cpu')
        torch.onnx.export(model, dummy_input, file.replace('.pkl', '.onnx'),
                        export_params=True, verbose=0, training=1, do_constant_folding=False)

        print(f"Done {file} ...")


gen_model_zoo('/mnt/zoo/vgg_zoo/')

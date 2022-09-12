import os

import torch
from vgg import *

conf_list = vgg_zoo()
dummy_input = torch.rand(8, 3, 32, 32) #  batch:32; 3 channels; 32 x 32 size
path = '/mnt/vgg100'

for idx, conf in enumerate(conf_list):
    model = VGG(make_layers(conf[0], k = conf[1]))
    num_params = sum(p.numel() for p in model.parameters())
    # torch.onnx.export(model, dummy_input, os.path.join(path, f"model_{idx}.onnx"), 
    #                     export_params=True, verbose=0, training=1)
    print(f"Successfully generate {idx}, {num_params}")

print("============")

import os

import torch
import torchvision.models as models


def is_valid_name(name):
    # 1. has numbers; 2. all lower case
    number = False
    for i in range(len(name)):
        if 'A' <= name[i] <= 'Z':
            return False
        if '1' <= name[i] <= '9':
            number = True
    return number


def gen_model_zoo(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

    args = dir(models)
    # batch:32; 3 channels; 32 x 32 size
    dummy_input = torch.rand(32, 3, 32, 32)

    for model_type in args:
        if is_valid_name(model_type):
            try:
                model = models.__dict__[model_type](num_classes=10)
                torch.onnx.export(
                    model,
                    dummy_input,
                    os.path.join(
                        path,
                        model_type +
                        ".onnx"),
                    export_params=True,
                    verbose=0,
                    training=1)
                print('Generate {} to zoo'.format(model_type))
            except Exception as e:
                print("Error: ", e)


gen_model_zoo('../zoo')

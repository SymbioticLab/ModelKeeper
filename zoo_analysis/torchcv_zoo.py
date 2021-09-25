from pytorchcv.model_provider import _models as model_zoo
from pytorchcv.model_provider import get_model as ptcv_get_model

import torch
from torch.autograd import Variable

import os 

def export_onnx():
	
	dummy_input = torch.rand(8, 3, 224, 224) #  batch:32; 3 channels; 32 x 32 size
	path = '/mnt/zoo/'
	cnt = 0
	params_set = set()

	for model_name in model_zoo:
		try:
			model = ptcv_get_model(model_name, pretrained=False)
			num_params = sum(p.numel() for p in model.parameters())

			if num_params in params_set:
				print(f"{model_name} is repeated")
				continue

			torch.onnx.export(model, dummy_input, os.path.join(path, model_name+".onnx"), 
	                          export_params=False, verbose=0, training=1)
			print(f"Successfully generate {model_name}, # of params {num_params}")

			params_set.add(num_params)
			cnt += 1

		except Exception as e:
			print(f"{model_name} failed due to {e}")

	print("============")
	print(f"Generate {cnt} models in total, failed {len(model_zoo)-cnt} models")


export_onnx()


from models.cifarmodels.model_provider import get_cv_model
import torch

def get_str_type(input_str):
    input_str = input_str.strip()

    if input_str.isdigit():
        return int(input_str)
    if input_str == 'True' or input_str == 'False':
        return eval(input_str)
    if '.' in input_str:
        return float(input_str)
    return str(input_str)

def get_model(temp_model_name):
    model_type = temp_model_name.split('(')[0].strip()
    temp_args = [x.strip() for x in temp_model_name.split('(')[1].replace(')','').strip().split(',') if len(x.strip())>0]
    args_model = {}
    for pair in temp_args:
        [_key, _value] = pair.split('=')

        args_model[_key.strip()] = get_str_type(_value)
        print(_value, type(get_str_type(_value)))

    args_model['name'] = model_type
    return args_model

tests = ['MobileNetV3(is_large=0, multiplier=0.75)', "MobileNetV3(is_large=1, multiplier=0.75)", "VGG(vgg_block=11)", "ShuffleNetG3()", "ShuffleNetV2(net_size=0.5)"]

for test in tests:
    args_model = get_model(test)
    net = get_cv_model(**args_model)
    print(net(torch.rand(2, 3, 32, 32)))

print('done..')


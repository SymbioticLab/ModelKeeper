import torch
from models.cifarmodels.model_provider import get_cv_model
from models.torchcv.model_provider import get_model as ptcv_get_model


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


tests = ["densenet250_k24_bc_cifar10", "densenet190_k40_bc_cifar10"]


for test in tests:
    temp_model_name = test 
    try:
        if '(' in temp_model_name:
            args_model = get_model(temp_model_name)
            #args_model['num_classes'] = num_classes

            model = get_cv_model(**args_model)
        else:
            model = ptcv_get_model(temp_model_name, pretrained=False)

        output = model(torch.rand(128, 3, 32, 32))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
        criteria = torch.nn.CrossEntropyLoss()
        loss = criteria(output, torch.zeros(128, dtype=torch.long))
        loss.backward()

    except Exception as e:
        print(f"{temp_model_name} failed... {e}" )

    #args_model = get_model(test)
    #net = get_cv_model(**args_model)
    

print('done..')

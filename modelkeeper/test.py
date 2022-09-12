import logging
import os

from matcher import ModelKeeper


def test():
    # import argparse

    # start_time = time.time()
    # zoo_path = '/mnt/zoo/tests/'

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--zoo_path', type=str, default=zoo_path)
    # parser.add_argument('--num_of_processes', type=int, default=30)
    # parser.add_argument('--neigh_threshold', type=float, default=0.05)

    # args = parser.parse_args()
    from config import modelkeeper_config
    zoo_path = '/users/fanlai/experiment/keeper/model_zoo'
    #zoo_path = "/users/fanlai/experiment/exp_logs/keeper/model_zoo/regnety002@0.4767.onnx"
    modelkeeper_config.zoo_path = zoo_path

    mapper = ModelKeeper(modelkeeper_config)

    #child_onnx_path = '/mnt/zoo/tests/vgg11.onnx'
    models = os.listdir(zoo_path)

    match_list = ['../query_zoo/nin_cifar10.onnx']
    for model in match_list:
        child_onnx_path = os.path.join(zoo_path, model)
        weights, meta_data = mapper.map_for_onnx(child_onnx_path, blacklist=set(
            [child_onnx_path]), model_name=child_onnx_path.split('/')[-1])

        logging.info(
            "\n\nMatching {}, results: {}\n".format(
                child_onnx_path, meta_data))

    # time.sleep(40)


test()
# test_fake()

def faked_graph():
    graph = nx.DiGraph(name='faked')
    attr1={'dims': [64, 32, 3, 3], 'op_type': 'cov1',}

    graph.add_node(0, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'0'})

    graph.add_node(1, attr={'dims': [2, 32, 3, 3], 'op_type': 'cov1', 'name':'1'})
    graph.add_node(2, attr={'dims': [2, 32, 3, 3], 'op_type': 'cov1', 'name':'2'})

    #graph.add_node(5, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'5'})
    #graph.add_node(3, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'3'})

    graph.add_node(4, attr={'dims': [5, 32, 3, 3], 'op_type': 'cov1', 'name':'4'})

    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 4)
    #graph.add_edge(0, 5)
    #graph.add_edge(5, 3)
    #graph.add_edge(3, 4)

    return graph

def faked_graph2():
    graph = nx.DiGraph(name='faked')
    attr1={'dims': [64, 32, 3, 3], 'op_type': 'cov1',}

    graph.add_node(0, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'0'})

    graph.add_node(1, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'1'})
    graph.add_node(2, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'2'})

    graph.add_node(5, attr={'dims': [2, 32, 3, 3], 'op_type': 'cov1', 'name':'5'})
    graph.add_node(3, attr={'dims': [2, 32, 3, 3], 'op_type': 'cov1', 'name':'3'})

    graph.add_node(4, attr={'dims': [5, 32, 3, 3], 'op_type': 'cov1', 'name':'4'})

    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 4)
    graph.add_edge(0, 5)
    graph.add_edge(5, 3)
    graph.add_edge(3, 4)

    return graph


def mapping_faked(parent, child_graph):
    opt = MatchingOperator(parent=parent)
    mappings, score = opt.get_mappings(child=child_graph)

    logging.info(opt.alignmentStrings()[0])
    logging.info("\n\n")
    logging.info(opt.alignmentStrings()[1])
    logging.info("\n\n")
    logging.info(opt.graphStrings()[0])
    logging.info("\n\n")
    logging.info(opt.graphStrings()[1])
    return (parent, mappings, score)

def test_fake():
    parent, child = faked_graph(), faked_graph2()

    mapper = MatchingOperator(parent=parent)
    logging.info(mapper.get_mappings(child))


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
    zoo_path = '/users/fanlai/experiment/temp_zoo'
    modelkeeper_config.zoo_path = zoo_path

    mapper = ModelKeeper(modelkeeper_config)

    #models = ["resnesta18@0.6394.onnx"]
    models = os.listdir(zoo_path)

    for model in models:
        child_onnx_path = os.path.join(zoo_path, model)
        weights, meta_data = mapper.map_for_onnx(child_onnx_path, blacklist=set([child_onnx_path]))

        logging.info("\n\nMatching {}, results: {}\n".format(child_onnx_path, meta_data))

    # time.sleep(40)

#test()
#test_fake()


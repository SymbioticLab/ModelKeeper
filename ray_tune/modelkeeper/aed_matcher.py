import pickle


class AEDMatcher(object):

    def __init__(self, path, name):
        self.name = name
        self.mappings = self.load_mapping(path)

    def load_mapping(self, path):
        with open(path, 'rb') as fin:
            stores = pickle.load(fin)

        return stores.get(self.name, None)

    def query_child(self, child):
        child = child.split('.onnx')[0]
        if self.mappings is None or child not in self.mappings:
            return -float('inf'), []

        dist = self.mappings[child]['GED']
        mappings = [(a,b) for a, b in zip(self.mappings[child]['Path'][0], self.mappings[child]['Path'][1])]
        return -dist, mappings

from clustering import k_medoids, k_medoids_auto_k

def load_data(file):
    model_pairs = {}
    num_of_nodes = 0
    with open(file) as fin:
        data = fin.readlines()
        data = [x for x in data if 'score' in x]
        parent_model = ''

        for line in data:
            if 'mapping pair' in line:
                parent_info, child_info = line.strip().split(',')
                parent_model = parent_info.split()[-1].replace('(','').replace('.onnx','')
                child_model = child_info.split()[0].split('/')[-1].replace('.onnx','').replace(')','')
                score = float(child_info.split()[-1])

                if parent_model not in model_pairs:
                    model_pairs[parent_model] = {}
                model_pairs[parent_model][child_model] = score

    return model_pairs


def get_dist_matrix(model_pairs):
    models = sorted(list(model_pairs.keys()))
    num_models = len(models)

    model_ids = {models[idx]:idx for idx in range(num_models)}
    dist_matrix = [[0] * num_models for _ in range(num_models)]

    for idx, model in enumerate(models):
        self_score = model_pairs[model][model]
        for i in range(num_models):
            child_model = models[i]
            dist_matrix[idx][i] = 1.0-model_pairs[model][child_model]/self_score

    return model_ids, dist_matrix 

model_pairs = load_data('/users/fanlai/nasbench201_scores')
#model_pairs = load_data('/users/fanlai/torchcv_scores')
model_ids, dist_matrix = get_dist_matrix(model_pairs)

def distance(a, b):
    return dist_matrix[a][b]

points = list(range(len(model_pairs)))
#diameter, medoids = k_medoids(points, k=30, distance=distance, spawn=2, max_iterations=10000)
#print(dist_matrix[0])
diameter, medoids = k_medoids_auto_k(points, distance=distance, spawn=1000, threads=40, verbose=True, 
                    diam_max=0.75, start_k=1, max_iterations=500000)


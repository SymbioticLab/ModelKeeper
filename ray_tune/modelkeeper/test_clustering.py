from clustering import k_medoids, k_medoids_auto_k
import pickle

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
            dist_matrix[idx][i] = max(1e-3, 1.0-(model_pairs[model][child_model]/self_score+model_pairs[child_model][model]/model_pairs[child_model][child_model])/2.0)
#            print(model, child_model, model_pairs[model][child_model]/self_score)

    return model_ids, dist_matrix


def distance(a, b):
    #print(f'= {a} {b}')
    return dist_matrix[a][b]

def get_oracle_dist(a):
    min_dist = float('inf')
    for i in range(len(dist_matrix[a])):
        if i != a:
            min_dist = min(min_dist, dist_matrix[a][i])
    return min_dist

class ModelQueryer(object):

    def __init__(self, medoids=None):
        self.medoids = {i: medoids[i] for i in range(len(medoids))}

    def search_best(self, model):

        # 1. Get the score of all medoids
        overhead = 0
        cluster_dis = []

        oracle_dist = get_oracle_dist(model)
        current_min_dist = [float('inf')]

        for medoid_id in self.medoids:
            medoid = self.medoids[medoid_id]
            dist = distance(model, medoid.kernel)
            cluster_dis.append((dist, medoid_id))

            current_min_dist.append(min(current_min_dist[-1], dist))

        
        # 2. Sort kernel distance
        cluster_dis.sort()

        # 3. Query different clusters until TTL

        for (dis, medoid_id) in cluster_dis:
            selected_medoid = self.medoids[medoid_id]

            # Query in-cluster nodes
            for element in selected_medoid.elements:
                if element != model:
                    current_min_dist.append(min(current_min_dist[-1], distance(model, element)))

        return current_min_dist, oracle_dist


def save_medoid(file, medoids, diameter):
    with open(file, 'wb') as fout:
        pickle.dump(medoids, fout)
        pickle.dump(diameter, fout)

def load_medoid(file):
    with open(file, 'rb') as fin:
        medoids = pickle.load(fin)
        diameter = pickle.load(fin)

    return medoids, diameter

file = 'transformer_zooscores_new'#'nasbench201_scores' #torchcv_scores_new
#model_pairs = load_data('/users/fanlai/nasbench201_scores')
model_pairs = load_data(f'/users/fanlai/{file}')
model_ids, dist_matrix = get_dist_matrix(model_pairs)

points = list(range(len(model_pairs)))
k = 1#int(len(points)**0.5)
spawn = 1 if k == 1 else 100000
medoid_path = f"{file}_medoids_{k}.pkl"

diameter, medoids = k_medoids(points, k=k, distance=distance, threads=40, spawn=spawn, max_iterations=10000)
save_medoid(medoid_path, medoids, diameter)
print(diameter, '\n', medoids)

# medoids, diameter = load_medoid(medoid_path)

queryer = ModelQueryer(medoids)
medoid_set = set([medoid.kernel for medoid in medoids])
query_points = [x for x in points if x not in medoid_set]

avg_query_line = []
for point in query_points:
    current_min_dist, oracle_dist = queryer.search_best(point)
    for i, x in enumerate(current_min_dist):
        if len(avg_query_line) <= i:
            avg_query_line.append([])
        avg_query_line[i].append(x/oracle_dist)

print([sum(x)/len(x) for x in avg_query_line])
#print(dist_matrix[0])
# diameter, medoids = k_medoids_auto_k(points, distance=distance, spawn=50000, threads=40, verbose=True,
#                      diam_max=1.0, start_k=1, max_iterations=500000)

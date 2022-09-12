import pickle

from clustering import k_medoids, k_medoids_auto_k

dist_matrix = {}


def load_data(file):
    model_pairs = {}

    with open(file) as fin:
        data = fin.readlines()
        data = [x for x in data if 'score' in x]

        for line in data:
            if 'align_child' in line:
                # parent_info, child_info = line.strip().split('onnx,')
                # child_model = parent_info.split()[-1].replace('(','').replace('.onnx','')
                # parent_model = child_info.split()[0].split('/')[-1].replace('.onnx','').replace(')','')
                # score = float(child_info.split()[-1])
                parent_model = line.split()[2].split('.')[0]
                child_model = line.split()[4].split('.')[0]
                score = float(line.split()[-1])
                if parent_model not in model_pairs:
                    model_pairs[parent_model] = {}
                model_pairs[parent_model][child_model] = score

    # normalize all scores
    keys = list(model_pairs.keys())
    for key in keys:
        try:
            temp = []
            for m in model_pairs:
                temp.append(model_pairs[m][key])
            temp.sort()
            _len = len(temp)
            _min, _max = temp[1], temp[-2]
            for m in model_pairs:
                model_pairs[m][key] = min(
                    max(1e-5, (model_pairs[m][key] - _min) / (_max - _min)), 0.9999)
        except Exception as e:
            print(e, key)
            del model_pairs[key]
    return model_pairs


def get_dist_matrix(model_pairs):
    models = sorted(list(model_pairs.keys()))
    num_models = len(models)

    model_ids = {models[idx]: idx for idx in range(num_models)}
    dist_matrix = [[0] * num_models for _ in range(num_models)]

    for idx, model in enumerate(models):
        self_score = model_pairs[model][model]
        for i in range(num_models):
            child_model = models[i]
            dist_matrix[idx][i] = max(1e-3, 1.0 -
                                      (model_pairs[model][child_model] +
                                       model_pairs[child_model][model]) /
                                      2.0)

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
        current_min_dist = [0]

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
                    current_min_dist.append(
                        min(current_min_dist[-1], distance(model, element)))

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


def dryrun(file='imagenet-cv-zoo', factor=1):
    global dist_matrix
    # file = 'imagenet-cv-zoo'#'nlp-zoo-score'#'imagenet-cv-zoo' #torchcv_scores_new
    #model_pairs = load_data('/users/fanlai/nasbench201_scores')
    model_pairs = load_data(f'./{file}')
    # print(len(model_pairs))
    model_ids, dist_matrix = get_dist_matrix(model_pairs)

    points = list(range(len(model_pairs)))
    k = int(len(points)**0.5 * factor + 1)
    spawn = 1 if k == 1 else 100000
    medoid_path = f"{file}_clustering_{k}.pkl"

    diameter, medoids = k_medoids(
        points, k=k, distance=distance, threads=40, spawn=spawn, max_iterations=10000)

    queryer = ModelQueryer(medoids)
    medoid_set = set([medoid.kernel for medoid in medoids])
    query_points = [x for x in points if x not in medoid_set]

    avg_query_line = []
    tiktak = []
    current_sim = []
    for point in query_points:
        current_min_dist, oracle_dist = queryer.search_best(point)
        not_skip = True
        for i, x in enumerate(current_min_dist):
            if len(avg_query_line) <= i:
                avg_query_line.append([])
                current_sim.append([])
            quality = (1. - x) / (1. - oracle_dist)
            avg_query_line[i].append(quality)
            current_sim[i].append(1. - x)

            if quality > 0.999 and not_skip:
                tiktak.append(i + 1)
                not_skip = False

    ans = [sum(x) / len(x) for x in avg_query_line]
    return {
        'Quality': ans,
        'Sim': [
            sum(x) / len(x) for x in current_sim],
        'Matches': sum(tiktak) / len(tiktak) / len(tiktak) * 100.0,
        'k': k}
    # print('Quality: ', ans)
    # print('Sim: ', [sum(x)/len(x) for x in current_sim])
    # print('# of Matches Taken: ', sum(tiktak)/len(tiktak)/len(tiktak)*100.0)


factors = [0, 0.2, 0.25, 0.33, 0.5, 1, 2, 3, 4, 5]
results = []
workload = 'imagenet-cv-zoo'
for f in factors:
    res = dryrun(workload, factor=f)
    results.append(res)
    print(res)

with open(workload + '.pkl', 'wb') as fout:
    pickle.dump(results, fout)

# with open(medoid_path, 'wb') as fout:
#     pickle.dump(ans, fout)
# print(dist_matrix[0])
# diameter, medoids = k_medoids_auto_k(points, distance=distance, spawn=50000, threads=40, verbose=True,
#                      diam_max=1.0, start_k=1, max_iterations=500000)

import pandas
import random
import csv

def gen_trace(model_list):
    # Pollux use workload 6
    workloads = ['workload-6.csv'] + [f'workload-{i}.csv' for i in range(1, 6)]

    arrival_trace = []
    base = 0

    # Concatenate multiple traces
    for w in workloads:
        workload = pandas.read_csv(w)
        for row in workload.sort_values(by="time").itertuples():
            if len(arrival_trace) > 0 and row.time+base < arrival_trace[-1]:
                base = arrival_trace[-1]
            arrival_trace.append(row.time+base)

    # Load model list
    random.seed(8)
    models = [x.strip() for x in open(model_list).readlines()]
    models_a = models[:12]
    models_b = models[12:]
    random.shuffle(models_b)
    models = models_a+models_b
    
    ans = [['name', 'arrival']]
    for arrival, model in zip(arrival_trace, models):
        ans.append([model, arrival])

    with open(f'{model_list}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(ans)

gen_trace('torchcv_list')

import pandas
import random
import csv

def gen_trace(model_list):
    # Pollux use workload 6
    workloads = ['workload-6.csv'] + [f'workload-{i}.csv' for i in range(1, 6)]

    arrival_trace = []
    base = 0
    factor = 8

    # Concatenate multiple traces
    for w in workloads:
        workload = pandas.read_csv(w)
        for row in workload.sort_values(by="time").itertuples():
            if len(arrival_trace) > 0 and row.time+base < arrival_trace[-1]:
                base = arrival_trace[-1]

            arrival_trace.append(row.time+base)
            
    # Load model list
    random.seed(0)
    models = [x.strip().split(':')[0] for x in open(model_list).readlines()]
    models_a = models[:8]
    models_b = models[8:]
    random.shuffle(models_b)
    models = models_a+models_b

    ans = [['name', 'arrival']]
    
    for arrival, model in zip(arrival_trace, models):
        ans.append([model, arrival*factor])

    with open(f"{model_list.split('/')[-1]}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(ans)

def shrink_trace(file):
    workload = pandas.read_csv(file)
    factor = 3
    ans = [['name', 'arrival']]
    for row in workload.itertuples():
        ans.append([row.name, int(row.arrival*factor)])

    with open(f"new_{file}", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(ans)

# gen_trace('../../zoo_analysis/nlp_nwp_zoo')
shrink_trace("nlp_nwp.csv")

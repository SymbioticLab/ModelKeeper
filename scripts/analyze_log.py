import os
import pandas

def load_files(path):
    res = {}
    dirs = os.listdir(path)
    for _dir in dirs:
        folder_path = os.path.join(path, _dir)
        if os.path.isdir(folder_path):
            model_name = _dir.split(", 'arrival':")[0].split()[-1]
            try:
                progress = pandas.read_csv(os.path.join(folder_path, "progress.csv"))
                is_success = False
                accuracy, total_time = [], []
                for row in progress.itertuples():
                    is_success = row.done or is_success
                    accuracy.append(row.mean_accuracy)
                    total_time.append(row.time_total_s)

                if is_success:
                    res[model_name] = {'accuracy': accuracy, 'time': total_time}
            except Exception as e:
                print(e)
                pass 

    return res

def scan_target(alist, target):
    for idx in range(len(alist)):
        if alist[idx] >= target:
            return idx+1

def get_target(alist, blist):
    target = min(max(alist), max(blist))
    return scan_target(alist, target), scan_target(blist, target)


baseline = load_files("TrainModel_2021-11-03_16-39-35")
keeper = load_files("TrainModel_2021-11-02_00-11-32")

factor = []
threshold = 10

for key in baseline:
    if key in keeper:
        epoch_a, epoch_b = get_target(baseline[key]['accuracy'], keeper[key]['accuracy'])
        if epoch_a > threshold and epoch_b > threshold:
            factor.append((epoch_a, epoch_b))

factor.sort(key=lambda x:x[0]/x[1])
print(factor)

factor = [x[0]/x[1] for x in factor]
print(sum(factor)/len(factor), len(factor))

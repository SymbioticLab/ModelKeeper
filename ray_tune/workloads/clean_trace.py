import csv

import pandas

file = 'torchcv_list.csv'

workload = pandas.read_csv(file)
lines = []
for row in workload.sort_values(by="arrival").itertuples():
    lines.append([row.name, row.arrival])

base = int(lines[0][1])
for i in range(len(lines)):
    lines[i][1] = int(lines[i][1]) - base

lines = [['name', 'arrival']] + lines
with open(f'new_{file}.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(lines)


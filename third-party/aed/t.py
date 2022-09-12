import pickle

with open('result.pkl', 'rb') as fout:
    result = pickle.load(fout)

durations = []
for m1 in result:
    for m2 in result[m1]:
        durations.append(result[m1][m2]['Duration'])

print(sum(durations)/len(durations))

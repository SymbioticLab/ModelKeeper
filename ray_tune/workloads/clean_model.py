
with open('modellist') as fin:
    lines = [x.split('(')[0] for x in fin.readlines()]

with open('modeldict', 'w') as fout:
    lines.sort()
    for line in lines:
        fout.writelines(f'"{line}": {line},\n')

import sys
import os
import pickle

with open('ray_cluster.conf', 'rb') as fin:
    ray_conf = pickle.load(fin)

master_port = ray_conf['master_port']
master_node = ray_conf['master_node']
master_address = ray_conf['master_ip']

with open('template.lsf', 'r') as fin:
    template_server = ''.join(fin.readlines())
    
with open('submit.lsf', 'w') as fout:
    scriptPS = template_server + '\n#BSUB -J submit\n#BSUB -e ray_job.e\n#BSUB -o ray_job.o\n' + '#BSUB -m "'+master_node+'"\n\n' 
    scriptPS += ("\npython ../ray_tune/ray_tuner.py " + ' '.join(sys.argv[1:]) + f' --address="{master_address}"\n')
    fout.writelines(scriptPS)

os.system('bsub < submit.lsf')

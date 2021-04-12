import sys, os, time, datetime
import random
import pickle

master_port = 6379 #random.randint(11000, 60000)
redis_port = 12345 #random.randint(11000, 60000)
master_node = 'gpu-cn004'

vm_gpus = {}
per_vm_gpu = 4
per_vm_cpu = 40
vm_gpus['gpu-cn008']=3
vm_gpus['gpu-cn013']=3

master_ip = "10.246.6." + str(int(master_node[-3:]) + 90) + ":"+str(master_port)

ray_conf = {}
ray_conf['master_node'] = master_node
ray_conf['master_port'] = master_port
ray_conf['master_ip'] = master_ip


with open('ray_cluster.conf', 'wb') as fout:
    pickle.dump(ray_conf, fout)


os.system("bhosts > vms")
os.system("rm *.o")
os.system("rm *.e")
os.system("rm core.*")

avaiVms = {}
quotalist = {}

with open('quotas', 'r') as fin:
    for v in fin.readlines():
        items = v.strip().split()
        quotalist[items[0]] = int(items[1])

threadQuota = 5

with open('vms', 'r') as fin:
    lines = fin.readlines()
    for line in lines:
        if 'gpu-cn0' in line:
            items = line.strip().split()

            status = items[1]
            threadsGpu = int(items[5])
            vmName = items[0]
            #print(vmName,'#', quotalist[vmName])
            maxQuota = quotalist[vmName] if vmName in quotalist else 999

            if status == "ok" and (40-threadsGpu) >= threadQuota and maxQuota >= threadQuota:
                avaiVms[vmName] = min(40 - threadsGpu, maxQuota)

# remove all log files, and scripts first
files = [f for f in os.listdir('.') if os.path.isfile(f)]

print(avaiVms)
for file in files:
    if 'worker' in file or 'head' in file:
        os.remove(file)
        
# get the number of workers first
numOfWorkers = int(sys.argv[1])

# load template
with open('template.lsf', 'r') as fin:
    template = ''.join(fin.readlines())

# load template
with open('template.lsf', 'r') as fin:
    template_server = ''.join(fin.readlines())

_time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m%d_%H%M%S')
timeStamp = str(_time_stamp) + '_'
jobPrefix = 'worker' + timeStamp


if master_node in avaiVms:
    avaiVms[master_node] -= threadQuota

assignedVMs = []
# assert(len(availGPUs) > numOfWorkers)

# generate new scripts, assign each worker to different vms
for w in range(1, numOfWorkers + 1):
    
    jobName = 'worker' + str(w)
    fileName = jobName#jobPrefix+str(w)


    _vm = sorted(avaiVms, key=avaiVms.get, reverse=True)[0]
    print('assign ...{} to {}'.format(str(w), _vm))
    assignedVMs.append(_vm)

    del avaiVms[_vm]

    assignVm = '\n#BSUB -m "{}"\n'.format(_vm)
    runCmd = template + assignVm + '\n#BSUB -J ' + jobName + '\n#BSUB -e ' + fileName + '.e\n'  + '#BSUB -o '+ fileName + '.o\n';
    runCmd += f'\nray stop \nray start --address="{master_ip}" --redis-password="5241590000000000" --num-cpus={per_vm_cpu} --num-gpus={vm_gpus.get(_vm, per_vm_gpu)} \nsleep 240h\n'

    with open('worker' + str(w) + '.lsf', 'w') as fout:
        fout.writelines(runCmd)

# deal with ps
rawCmdPs = f"\nray stop \nray start --head --num-cpus=1 --num-gpus=0 --redis-port={master_port} --redis-shard-ports={master_port+1} --node-manager-port={redis_port} --object-manager-port={redis_port+1} \nsleep 240h\n"

with open('master.lsf', 'w') as fout:
    scriptPS = template_server + '\n#BSUB -J master\n#BSUB -e master.e\n#BSUB -o master.o\n' + '#BSUB -m "'+master_node+'"\n\n' + rawCmdPs
    fout.writelines(scriptPS)

os.system('bsub < master.lsf')

time.sleep(5)
os.system('rm vms')

vmSets = set()
for w in range(1, numOfWorkers + 1):
    time.sleep(1)
    vmSets.add(assignedVMs[w-1])
    os.system('bsub < worker' + str(w) + '.lsf')

os.system('rm worker*.lsf')
os.system('rm master.lsf')

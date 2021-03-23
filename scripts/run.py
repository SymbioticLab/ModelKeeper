import sys, os, time, datetime, random

master_node = 'gpu-cn001'
paramsCmd = ' --num_models 4 '
master_ip = "10.246.6." + str(int(master_node[-3:]) + 90) + ":6379"

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
# learner = ' --learners=1'

# for w in range(2, numOfWorkers+1):
#     learner = learner + '-' + str(w)

# load template
with open('template.lsf', 'r') as fin:
    template = ''.join(fin.readlines())

# load template
with open('template.lsf', 'r') as fin:
    template_server = ''.join(fin.readlines())

_time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m%d_%H%M%S')
timeStamp = str(_time_stamp) + '_'
jobPrefix = 'worker' + timeStamp

# get the join of parameters
params = ' '.join(sys.argv[2:]) +  ' '

rawCmd = '\npython ~/Kuiper/learner.py' + paramsCmd


if master_node in avaiVms:
    avaiVms[master_node] -= threadQuota

assignedVMs = []
# assert(len(availGPUs) > numOfWorkers)

# generate new scripts, assign each worker to different vms
for w in range(1, numOfWorkers + 1):
    
    fileName = jobPrefix+str(w)
    jobName = 'worker' + str(w)

    _vm = sorted(avaiVms, key=avaiVms.get, reverse=True)[0]
    print('assign ...{} to {}'.format(str(w), _vm))
    assignedVMs.append(_vm)

    del avaiVms[_vm]

    assignVm = '\n#BSUB -m "{}"\n'.format(_vm)
    runCmd = template + assignVm + '\n#BSUB -J ' + jobName + '\n#BSUB -e ' + fileName + '.e\n'  + '#BSUB -o '+ fileName + '.o\n';
    runCmd += "\nray stop \nray start --address='" + master_ip + "'  --redis-password='5241590000000000'\nsleep 24h\n" 

    with open('worker' + str(w) + '.lsf', 'w') as fout:
        fout.writelines(runCmd)

# deal with ps
rawCmdPs = "\nray stop \nray start --head --redis-port=6379 --redis-shard-ports=6380 --node-manager-port=12345 --object-manager-port=12346 \nsleep 24h\n"

with open('head.lsf', 'w') as fout:
    scriptPS = template_server + '\n#BSUB -J head\n#BSUB -e head{}'.format(timeStamp) + '.e\n#BSUB -o head{}'.format(timeStamp) + '.o\n' + '#BSUB -m "'+master_node+'"\n\n' + rawCmdPs
    fout.writelines(scriptPS)

with open('submit.lsf', 'w') as fout:
    scriptPS = template_server + '\n#BSUB -J submit\n#BSUB -e submit{}'.format(timeStamp) + '.e\n#BSUB -o submit{}'.format(timeStamp) + '.o\n' + '#BSUB -m "'+master_node+'"\n\n' 
    scriptPS += "\npython ../ray_tune/ray_tuner.py\n"
    fout.writelines(scriptPS)

os.system('bsub < head.lsf')

time.sleep(5)
os.system('rm vms')

vmSets = set()
for w in range(1, numOfWorkers + 1):
    time.sleep(5)
    vmSets.add(assignedVMs[w-1])
    os.system('bsub < worker' + str(w) + '.lsf')

time.sleep(10)
os.system('bsub < submit.lsf')

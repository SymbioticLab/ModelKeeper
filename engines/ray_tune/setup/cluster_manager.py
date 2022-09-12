# Submit job to the remote cluster

import datetime
import os
import pickle
import random
import subprocess
import sys
import time

import yaml


def load_yaml_conf(yaml_file):
    with open(yaml_file) as fin:
        data = yaml.load(fin, Loader=yaml.FullLoader)
    return data

def process_cmd(yaml_file):

    yaml_conf = load_yaml_conf(yaml_file)

    master_ip = yaml_conf['master_ip']
    worker_ips, total_gpus = [], []
    cmd_script_list = []

    executor_configs = ";".join(yaml_conf['worker_ips'])
    for ip_gpu in yaml_conf['worker_ips']:
        ip, num_gpu = ip_gpu.strip().split(':')
        worker_ips.append(ip)
        total_gpus.append(num_gpu)

    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m%d_%H%M%S')
    running_vms = set()
    job_name = 'modelkeeper'
    log_path = './logs'
    submit_user = f"{yaml_conf['auth']['ssh_user']}@" if len(yaml_conf['auth']['ssh_user']) else ""

    job_conf = {'time_stamp':time_stamp,
                'master_ip':master_ip,
                }

    for conf in yaml_conf['job_conf']:
        job_conf.update(conf)

    conf_script = ''
    setup_cmd = ''
    if yaml_conf['setup_commands'] is not None:
        setup_cmd += (yaml_conf['setup_commands'][0] + ' && ')
        for item in yaml_conf['setup_commands'][1:]:
            setup_cmd += (item + ' && ')

    cmd_sufix = f" "


    for conf_name in job_conf:
        conf_script = conf_script + f' --{conf_name}={job_conf[conf_name]}'
        if conf_name == "job_name":
            job_name = job_conf[conf_name]
        if conf_name == "log_path":
            log_path = os.path.join(job_conf[conf_name], 'log', job_name, time_stamp)

    # =========== Submit job to parameter server ============
    running_vms.add(master_ip)
    ps_cmd = f'ray start --head --address={master_ip}:6379 --redis-password="5241590000000000" --num-cpus=1 --num-gpus=0  && sleep 240h'
    
    with open(f"{job_name}_logging", 'wb') as fout:
        pass

    print(f"Starting master on {master_ip}...")
    with open(f"{job_name}_logging", 'a') as fout:
        subprocess.Popen(f'ssh {submit_user}{master_ip} "{setup_cmd} {ps_cmd}"',
                        shell=True, stdout=fout, stderr=fout)

    time.sleep(5)
    # =========== Submit job to each worker ============
    rank_id = 1
    for worker, n_gpu in zip(worker_ips, total_gpus):
        running_vms.add(worker)
        print(f"Starting workers on {worker} ...")

        worker_cmd = f'ray start --address={master_ip}:6379 --redis-password="5241590000000000" --num-cpus={10} --num-gpus={n_gpu} && sleep 240h'
        
        with open(f"{job_name}_logging", 'a') as fout:
            time.sleep(0.5)
            subprocess.Popen(f'ssh {submit_user}{worker} "{setup_cmd} {worker_cmd}"',
                            shell=True, stdout=fout, stderr=fout)

    # dump the address of running workers
    current_path = os.path.dirname(os.path.abspath(__file__))
    job_name = os.path.join(current_path, job_name)
    with open(job_name, 'wb') as fout:
        job_meta = {'user':submit_user, 'vms': running_vms}
        pickle.dump(job_meta, fout)

    print(f"Submitted job, please check your logs ({log_path}) for status")


def terminate(job_name):

    current_path = os.path.dirname(os.path.abspath(__file__))
    job_meta_path = os.path.join(current_path, job_name)

    if not os.path.isfile(job_meta_path):
        print(f"Fail to terminate {job_name}, as it does not exist")

    with open(job_meta_path, 'rb') as fin:
        job_meta = pickle.load(fin)

    for vm_ip in job_meta['vms']:
        # os.system(f'scp shutdown.py {job_meta["user"]}{vm_ip}:~/')
        print(f"Shutting down job on {vm_ip}")
        with open(f"{job_name}_logging", 'a') as fout:
            subprocess.Popen(f'ssh {job_meta["user"]}{vm_ip} "ray stop; killall sleep"',
                            shell=True, stdout=fout, stderr=fout)


if sys.argv[1] == 'init':
    process_cmd(sys.argv[2])
elif sys.argv[1] == 'stop':
    terminate(sys.argv[2])
else:
    print("Unknown cmds ...")

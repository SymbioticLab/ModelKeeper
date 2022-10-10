## Install the dependency 

You can simply run `install.sh`.

```
conda env create -f environment. yml
```

## Setting Up GPU Cluster

**Note:**
Please assure that these paths are consistent across all nodes so that ModelKeeper can find the right path.

- ***Coordinator node***: Make sure that the coodinator (master node) has access to other worker nodes via ```ssh```. 

- ***All nodes***: Follow the same dependency setup.

## Running the experiment
```
python keeper_nasbench201_graph.py --use_keeper --user=xxx --num_nodes=xxx --num_gpu_per_nodes=xxx --max_trial_number_per_gpu=xxx --max_trial_number=xxx
```
import argparse

parser = argparse.ArgumentParser()

# Path configuration
parser.add_argument('--exe_path', type=str, default='./', help='Data store of the framework')
parser.add_argument('--zoo_path', type=str, default='/gpfs/gpfs0/groups/chowdhury/fanlai/model_zoo/nasbench201', help='Path of the model zoo')

# Framework configuration
parser.add_argument('--num_of_processes', type=int, default=4, help='Number of threads used for mapping (~CPU cores)')

# Parameters
parser.add_argument('--neigh_threshold', type=float, default=0.9, 
                    help='Threshold of evicting neighbors a if score(a,b)<T and acc(a)<acc(b)')

oort_config, unknown = parser.parse_known_args()

import argparse

parser = argparse.ArgumentParser()

# Path configuration
parser.add_argument('--zoo_path', type=str, default='/users/fanlai/experiment/keeper/model_zoo/', help='Path of the model zoo')
parser.add_argument('--execution_path', type=str, default='/users/fanlai/experiment/keeper/jobs/', help='Runtime data store of the framework')
parser.add_argument('--zoo_query_path', type=str, default='/users/fanlai/experiment/keeper/query_zoo/', help='Runtime data store of querying models')
parser.add_argument('--zoo_ans_path', type=str, default='/users/fanlai/experiment/keeper/ans_zoo/', help='Runtime data store of querying results')
parser.add_argument('--zoo_register_path', type=str, default='/users/fanlai/experiment/keeper/register_zoo/', help='Runtime data store of new pending models')

# Framework configuration
parser.add_argument('--num_of_processes', type=int, default=20, help='Number of threads used for mapping (~CPU cores)')
parser.add_argument('--zoo_server', type=str, default='10.0.0.1', help='Server of ModelKeeper')
parser.add_argument('--user_name', type=str, default='', help='User name in accessing the ModelKeeper server')

# Parameters
parser.add_argument('--neigh_threshold', type=float, default=0.1,
                    help='Threshold of evicting neighbors a if score(a,b)<T and acc(a)<acc(b)')

modelkeeper_config, unknown = parser.parse_known_args()

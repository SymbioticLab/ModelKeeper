import argparse

parser = argparse.ArgumentParser()

# Path configuration
parser.add_argument('--zoo_path', type=str, default='./model_zoo/', help='Path of the model zoo')
parser.add_argument('--execution_path', type=str, default='./jobs/', help='Runtime data store of the framework')
parser.add_argument('--zoo_query_path', type=str, default='./query_zoo/', help='Runtime data store of querying models')
parser.add_argument('--zoo_ans_path', type=str, default='./ans_zoo/', help='Runtime data store of querying results')


# Framework configuration
parser.add_argument('--num_of_processes', type=int, default=4, help='Number of threads used for mapping (~CPU cores)')
parser.add_argument('--zoo_server', type=str, default='10.0.0.1', help='Server of ModelKeeper')
parser.add_argument('--user_name', type=str, default='', help='User name in accessing the ModelKeeper server')

# Parameters
parser.add_argument('--neigh_threshold', type=float, default=0.9, 
                    help='Threshold of evicting neighbors a if score(a,b)<T and acc(a)<acc(b)')

modelkeeper_config, unknown = parser.parse_known_args() 

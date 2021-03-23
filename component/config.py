import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--exe_path', type=str, default='./', help='Data path of the framework')
parser.add_argument('--num_of_processes', type=int, default=16, help='Number of threads used for mapping (~CPU cores)')

args = parser.parse_args()

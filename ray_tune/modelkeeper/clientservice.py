import os
import time
import pickle
from paramiko import SSHClient
from scp import SCPClient

class ModelKeeperClient(object):

    """A very simple client service for ModelKeeper"""

    def __init__(self, args):

        self.zoo_server = args.zoo_server

        # TODO: These paths should be informed after querying the zoo host
        self.zoo_path = args.zoo_path
        self.zoo_query_path = args.zoo_query_path
        self.zoo_ans_path = args.zoo_ans_path
        self.zoo_register_path = args.zoo_register_path

        self.execution_path = args.execution_path

        self.connection = self.create_connection()
        self.connection_manager = SCPClient(self.connection.get_transport())

    def create_connection(self):
        connection = SSHClient()
        connection.load_system_host_keys()
        connection.connect(self.zoo_server)

        return connection

    def query_for_model(self, model_path, timeout=240):
        """
        @ model: assume the model is in onnx format
        """
        model_name = model_path.split('/')[-1] # xx.onnx
        ans_model_name = model_name + '.pkl'
        local_path = os.path.join(self.execution_path, ans_model_name)
        
        # 1. Upload the model to the modelkeeper pending queue
        self.register_model_to_zoo(model_path, self.zoo_query_path)

        # 2. Ping the host for results
        waiting_duration = 0 
        os.system(f'echo > {local_path}')

        while waiting_duration < timeout:
            success = self.pull_model_from_zoo(os.path.join(self.zoo_ans_path, ans_model_name), local_path)
            if not success:
                time.sleep(10)
                waiting_duration += 10
            else: break

        # 3. Remove result file from remote
        weights = meta = None

        if waiting_duration < timeout:
            # 3. Load model weights and return weights
            with open(local_path, 'rb') as fin:
                weights = pickle.load(fin)
                meta = pickle.load(fin) # {"matching_score", "parent_name", "parent_acc"}
        else:
            print(f"Querying the zoo server times out {timeout} sec")

        return weights, meta


    def register_model_to_zoo(self, model_path, zoo_path=None):
        """
        @ model: upload the model to the ModelKeeper zoo
        """
        if zoo_path is None:
            zoo_path = self.zoo_register_path

        try:
            self.connection_manager.put(model_path, zoo_path)
            logging.warning(f"Successfully upload model {model_path} to the zoo server")
        except Exception as e:
            logging.warning(f"Failed to connect to the zoo host {self.zoo_server}")
        

    def pull_model_from_zoo(self, model_path, local_path):
        """
        @ return the warmed weights of model
        """
        success = True
        try:
            self.connection_manager.get(model_path, local_path)
            stdin,stdout,stderr = self.connection.exec_command(f"rm {model_path}", timeout=30)
            stdout.channel.recv_exit_status()
        except Exception as e:
            success = False

        return success

    def stop(self):
        self.connection_manager.close()
        self.connection.close()
        os.rmdir(self.execution_path)


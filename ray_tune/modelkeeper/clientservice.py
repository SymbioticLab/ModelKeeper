import os
import time
import pickle


class ModelKeeperClient(object):

    """A very simple client service for ModelKeeper"""

    def __init__(self, args):
        self.args = args
        self.user_name = self.args.user_name
        self.zoo_server = self.args.zoo_server

        # TODO: These paths should be informed after querying the zoo host
        self.zoo_path = self.args.zoo_path
        self.zoo_query_path = self.args.zoo_query_path
        self.zoo_ans_path = self.args.zoo_ans_path

        self.execution_path = self.args.execution_path


    def query_for_model(self, model_path, timeout=240):
        """
        @ model: assume the model is in onnx format
        """
        model_name = model_path.split('/')[-1] # xx.onnx
        ans_model_name = model_name + '.pkl'
        
        # 1. Upload the model to the modelkeeper pending queue
        self.register_model_to_zoo(model_path, self.zoo_query_path)

        # 2. Ping the host for results
        waiting_duration = 0 

        while waiting_duration < timeout:
            status = self.pull_model_from_zoo(os.path.join(self.zoo_ans_path, ans_model_name))
            if status != 0:
                time.sleep(10)
                waiting_duration += 10
            else: break

        weights = meta = None

        if waiting_duration < timeout:
            # 3. Load model weights and return weights
            with open(os.path.join(self.execution_path, ans_model_name), 'rb') as fin:
                weights = pickle.load(fin)
                meta = pickle.load(fin) # {"matching_score", "parent_name", "parent_acc"}
        else:
            print(f"Querying the zoo server times out {timeout} sec")

        return weights, meta


    def register_model_to_zoo(self, model_path, zoo_path):
        """
        @ model: upload the model to the ModelKeeper zoo
        """
        status = os.system(f"scp {model_path} {self.user_name}@{self.zoo_server}:{zoo_path}")
        assert (status == 0, f"Failed to connect to the zoo host {self.zoo_server}")
        print(f"Successfully upload model {model_path} to the zoo server")


    def pull_model_from_zoo(self, model_path):
        """
        @ return the warmed weights of model
        """
        status = os.system(f"scp {self.user_name}@{self.zoo_server}:{model_path} {self.execution_path}")

        return status



import json
import os
import numpy as np


class Environment:
    '''
    Simulates NAS environment. Architecutres can be trained for a specified amount of epochs. 
    Tarining results are cached, that is, training the same model for larer epochs 
    will be timed as a continuation from the model's checkpoint. 
    '''
    def __init__(self, logs_dir):
        self._logs = []
        self._arch_to_id = {}
        
        arch_id = 0
        for i, filename in enumerate(os.listdir(logs_dir)):
            if filename.endswith('.json'):
                log_path = os.path.join(logs_dir, filename)
                x = json.load(open(log_path, 'r'))
                self._logs.append(x)
                assert x['recepie'] not in self._arch_to_id
                self._arch_to_id[x['recepie']] = arch_id
                arch_id += 1
        
        self._training_states = {}
        
    def get_total_time(self):
        return sum([x['wall_time'] for x in self._training_states.values()])
    

    def get_best_possible_test_loss(self):
        min_loss = np.inf
        for log in self._logs:
            if len(log['test_losses']) > 0:
                cur_loss = np.nanmin(log['test_losses'])
                if cur_loss < min_loss:
                    min_loss = cur_loss
        return min_loss

    def get_test_loss_of_the_best_validated_architecture(self):
        return self._logs[self.best_arch_id]['test_losses'][self.best_arch_epoch]
            
    def get_precomputed_recepies(self):
        return [json.loads(x['recepie']) for x in self._logs]
    
    def get_recepie_ids(self):
        return [x['recepie_id'] for x in self._logs]
    
    def reset(self):
        self.best_arch_id = -1
        self.best_arch_epoch = -1
        self._training_states = {}
        
    def _make_state_dict(self, arch_id, epoch):
        state_dict = {f'{phase}_loss':self._logs[arch_id][f'{phase}_losses'][epoch] if epoch >= 0 else np.nan 
                      for phase in ['train', 'val', 'test']}
        state_dict['wall_time'] = np.sum(self._logs[arch_id]['wall_times'][:epoch])
        state_dict['cur_epoch'] = epoch
        state_dict['status'] = 'OK' if epoch < len(self._logs[arch_id]['train_losses']) - 1 else self._logs[arch_id]['status']
        return state_dict
    
    def simulated_train(self, arch, max_epoch):
        arch_id = self._arch_to_id[json.dumps(arch)]
        if (arch_id not in self._training_states) or (max_epoch > self._training_states[arch_id]['cur_epoch']):
            max_epoch = min([max_epoch, len(self._logs[arch_id]['train_losses']) - 1])
            self._training_states[arch_id] = self._make_state_dict(arch_id, max_epoch)
            
            # update best result
            val_losses = self._logs[arch_id]['val_losses'][:self._training_states[arch_id]['cur_epoch'] + 1]
            if np.sum(~np.isnan(val_losses)) > 0:
                cur_best_epoch = np.nanargmin(val_losses)
                if (self.best_arch_id == -1) or\
                (self._logs[self.best_arch_id]['val_losses'][self.best_arch_epoch] > val_losses[cur_best_epoch]):
                    self.best_arch_id = arch_id
                    self.best_arch_epoch = cur_best_epoch
        
    def get_model_status(self, arch):
        arch_id = self._arch_to_id[json.dumps(arch)]
        return self._training_states[arch_id]['status']
    
    def get_model_stats(self, arch, epoch):
        arch_id = self._arch_to_id[json.dumps(arch)]
        if self._training_states[arch_id]['cur_epoch'] < epoch:
            raise Exception('Required epoch exceeds current training epochs.')
        
        return self._make_state_dict(arch_id, epoch)
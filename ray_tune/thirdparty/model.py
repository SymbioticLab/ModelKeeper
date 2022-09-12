import json

import numpy as np
import torch
import torch.nn
from thirdparty.custom_rnn import CustomRNN
from thirdparty.embed_regularize import embedded_dropout
from thirdparty.locked_dropout import LockedDropout
from thirdparty.weight_drop import ParameterListWeightDrop, WeightDrop


class AWDRNNModel(torch.nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, 
                 dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False,
                 recepie=None, verbose=True):
        super(AWDRNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = torch.nn.Dropout(dropouti)
        self.hdrop = torch.nn.Dropout(dropouth)
        self.drop = torch.nn.Dropout(dropout)
        self.encoder = torch.nn.Embedding(ntoken, ninp)
        self.wdrop = wdrop
        self.verbose = verbose
        
        if recepie is not None:
            recepie = json.loads(recepie)
            
        self.rnns = []
        for i in range(nlayers):
            input_size = ninp if i == 0 else nhid
            hidden_size = nhid if i != nlayers - 1 else (ninp if tie_weights else nhid)
            if rnn_type == 'LSTM':
                self.rnns.append(torch.nn.LSTM(input_size, hidden_size))
            elif rnn_type == 'CustomRNN':
                self.rnns.append(CustomRNN(input_size, hidden_size, recepie))

        if wdrop:
            if rnn_type == 'LSTM':
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
            elif rnn_type == 'CustomRNN':
                wd_rnns = []
                for rnn in self.rnns:
                    multilinear_components = []
                    for k, v in rnn.cell.components.items():
                        if rnn.cell.recepie[k]['op'] == 'linear':
                            for i in np.where(np.array(rnn.cell.recepie[k]['input']) != 'x')[0]:
                                multilinear_components.append(f'cell.components.{k}.weights.{i}')
                    wd_rnns.append(ParameterListWeightDrop(rnn, multilinear_components, dropout=wdrop))
                    self.rnns = wd_rnns
       
        if self.verbose:
            print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = torch.nn.Linear(nhid, ntoken)

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights
        self.recepie = recepie
        
    def reset(self):
        pass

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for i, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[i])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if i != self.nlayers - 1:
                #self.hdrop(raw_output) add??? 
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)
        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        hidden = []
        for i in range(self.nlayers):
            if self.rnn_type == 'LSTM':
                hidden_tuple_size = 2
            elif self.rnn_type == 'CustomRNN':
                if self.wdrop: 
                    # wrapped with ParameterListWeightDrop
                    hidden_tuple_size = self.rnns[0].module.cell.hidden_tuple_size
                else:
                    hidden_tuple_size = self.rnns[0].cell.hidden_tuple_size
            hidden_size = self.nhid if i != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)
            hidden.append(tuple([weight.new(1, bsz, hidden_size).zero_() for _ in range(hidden_tuple_size)]))
        
        return hidden    
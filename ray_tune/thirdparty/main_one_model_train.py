import torch
import torch.nn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from splitcross import SplitCrossEntropyLoss

import numpy as np
import networkx as nx
import math
import json
import time

import data
import os
from utils import batchify
from argparse import Namespace
from model import AWDRNNModel
from train import train, evaluate
import datetime

import argparse

parser = argparse.ArgumentParser(description='PyTorch Custom RNN Language Model')

parser.add_argument('--dataset_path', type=str, default='data/ptb',
                    help='location of the data corpus')
parser.add_argument('--logs_path', type=str, default='tmp',
                    help='path to logs folder')
parser.add_argument('--recepies_list_path', type=str, default='data/recepies_example.json',
                    help='list of models recepies')
parser.add_argument('--recepie_id', type=int, required=True,
                    help='id of a model recepie from the models list')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train')
parser.add_argument('--emsize', type=int, default=400,
                    help='emsize')
parser.add_argument('--nhid', type=int, default=600,
                    help='nhid')
parser.add_argument('--nlayers', type=int, default=3,
                    help='nlayers')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout')
parser.add_argument('--dropouth', type=float, default=0.25,
                    help='dropouth')
parser.add_argument('--dropouti', type=float, default=0.4,
                    help='dropouti')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropoute')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='wdrop')
parser.add_argument('--experiment_id', type=int,
                    help='some specific id of the experiment')

if __name__ == '__main__':
    
    init_time = str(datetime.datetime.now()).replace(':', '-').split('.')[0].replace(' ', '_')
    
    main_args = parser.parse_args()
    
    if main_args.experiment_id is None:
        main_args.experiment_id = 999999999 - np.random.randint(100000)
    
    all_recepies = json.load(open(main_args.recepies_list_path, 'r'))
    
    args = Namespace(data=main_args.dataset_path,
                     recepie_id=main_args.recepie_id,
                     recepies_list_path=main_args.recepies_list_path,
                     cuda=True,
                     batch_size=20,
                     model='CustomRNN',
                     emsize=main_args.emsize,
                     nhid=main_args.nhid, 
                     nlayers=main_args.nlayers,
                     dropout=main_args.dropout,
                     dropouth=main_args.dropouth,
                     dropouti=main_args.dropouti,
                     dropoute=main_args.dropoute,
                     wdrop=main_args.wdrop,
                     tied=True,
                     bptt=70,
                     lr=1e-3,
                     wdecay=1.2e-6,
                     epochs=main_args.epochs,
                     alpha=2,
                     beta=1,
                     log_interval=200,
                     clip=0.25,
                     eval_batch_size = 50,
                     recepie=json.dumps(all_recepies[main_args.recepie_id]))
    
    corpus = data.Corpus(args.data)
    cuda = 'cuda'

    train_data = batchify(corpus.train, args.batch_size, args, cuda)
    train_eval_data = batchify(corpus.train, args.eval_batch_size, args, cuda)
    val_data = batchify(corpus.valid, args.eval_batch_size, args, cuda)
    test_data = batchify(corpus.test, args.eval_batch_size, args, cuda)
    
    ntokens = len(corpus.dictionary)
    
    custom_model = AWDRNNModel(args.model, 
                               ntokens, 
                               args.emsize, 
                               args.nhid, 
                               args.nlayers, 
                               args.dropout, 
                               args.dropouth, 
                               args.dropouti, 
                               args.dropoute, 
                               args.wdrop, 
                               args.tied,
                               args.recepie,
                               verbose=False)
    
    
    log_stats = vars(args)
    log_stats['experiment_id'] = main_args.experiment_id
    log_stats['init_time'] = init_time
    log_stats['num_params'] = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] 
                                  for x in custom_model.parameters() if x.size())
    
    
    criterion = SplitCrossEntropyLoss(args.emsize, splits=[], verbose=False)
    
    if args.cuda:
        custom_model = custom_model.to(cuda)
        criterion = criterion.to(cuda)

    params = list(custom_model.parameters()) + list(criterion.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

    lr = args.lr
    train_losses = []
    val_losses = []
    test_losses = []
    wall_times = []

    # At any point you can hit Ctrl + C to break out of training early.
    status = 'OK'
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train(custom_model, optimizer, params, criterion, train_data, args, epoch)
            epoch_end_time = time.time()
            train_loss = evaluate(custom_model, criterion, train_eval_data, args.eval_batch_size, args)
            val_loss = evaluate(custom_model, criterion, val_data, args.eval_batch_size, args)
            test_loss = evaluate(custom_model, criterion, test_data, args.eval_batch_size, args)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s |\n| train loss {:5.2f} | '
                'train ppl {:8.2f} | train bpw {:8.3f} |\n| valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpw {:8.3f} |\n| test loss {:5.2f} | '
                'test ppl {:8.2f} | test bpw {:8.3f} |'.format(
              epoch, (epoch_end_time - epoch_start_time), 
                    train_loss, math.exp(train_loss), train_loss / math.log(2),
                    val_loss, math.exp(val_loss), val_loss / math.log(2),
                test_loss, math.exp(test_loss), test_loss / math.log(2)))
            print('-' * 89)

            wall_times.append(epoch_end_time - epoch_start_time)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)
            
            if np.isnan(np.array([train_loss, val_loss, test_loss])).any():
                status = 'loss is nan!'
                break

    except KeyboardInterrupt:
        print('-' * 89)
        status = 'KeyboardInterrupt'
        print('Exiting from training early')
    except Exception as e:
        status = 'Exception: ' + str(e)
        print('Exception', e)

    log_stats['wall_times'] = wall_times
    log_stats['train_losses'] = train_losses
    log_stats['val_losses'] = val_losses
    log_stats['test_losses'] = test_losses
    log_stats['status'] = status
    
    json.dump(log_stats, open(os.path.join(main_args.logs_path, f'log_stats_model_{args.recepie_id}_{init_time}_{main_args.experiment_id}.json'), 'w'))
    torch.save(custom_model.state_dict(), os.path.join(main_args.logs_path, f'dump_weights_model_{args.recepie_id}_{init_time}_{main_args.experiment_id}.pt'))




    
    
    
    
    
    
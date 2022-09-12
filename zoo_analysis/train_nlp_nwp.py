import logging
import os
import pickle

import torch
import torchtext
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

wiki = torchtext.datasets.WikiText103(root='~/experiment', split='train')
max_text_length = 256
tokenizer = None

path = '/users/fanlai/experiment/nwp_zoo'

device = 'cpu'

def collate(examples):
    if tokenizer._pad_token is None:
        return pad_sequence(examples, batch_first=True)
    return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

def tokenize_datset(tokenizer, data, block_size=256):

    batch_encoding = tokenizer([x for x in data if len(x)>0 and not x.isspace()], 
                        add_special_tokens=True, truncation=True, max_length=block_size)
    examples = batch_encoding["input_ids"]
    data = []
    max_len = 0
    for e in examples:
        data.append({"input_ids": torch.tensor(e, dtype=torch.long)})
        max_len = max(max_len, len(e))

    return data, max_len + 8

def eval_nwp(model, test_loader):
    total_loss = 0
    model.eval()

    for inputs in test_loader:
        inputs = inputs.to(device=device)
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss

        total_loss += loss.item()

    print(f"Eval loss: {total_loss/len(test_loader)}")

def train_nwp(model_name):
    global tokenizer
    pure_name = model_name.replace('/', '_')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_config(config)
    #model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer.max_length = max_text_length

    if not os.path.exists(f'{pure_name}_train') and not os.path.exists(f'{pure_name}_test'):
        train_dataset, max_len_train = tokenize_datset(tokenizer, torchtext.datasets.WikiText103(root='~/experiment', split='train'))
        test_dataset, max_len_test = tokenize_datset(tokenizer, torchtext.datasets.WikiText103(root='~/experiment', split='test'))

        with open(f'{pure_name}_train', 'wb') as fout:
            pickle.dump(train_dataset, fout)

        with open(f'{pure_name}_test', 'wb') as fout:
            pickle.dump(test_dataset, fout)
    else:
        with open(f'{pure_name}_train', 'rb') as fin:
            train_dataset = pickle.load(fin)
        with open(f'{pure_name}_test', 'rb') as fin:
            test_dataset = pickle.load(fin)

    max_text_length_model = max(max_len_train, max_len_test)
    config = AutoConfig.from_pretrained(model_name)
    config.max_length = max_text_length_model
    config.max_position_embeddings = max_text_length_model
    model = AutoModelForMaskedLM.from_config(config)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                    num_workers=4, collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, 
                    num_workers=4, collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15))

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=4e-5, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, verbose=True, min_lr=1e-6, factor=0.5)

    EPOCHS = 1
    step_interval = 100
    cur_step = 0

    model = model.to(device=device)

    for epoch in range(EPOCHS):
        total_loss = last_loss = cur_step = 0
        model.train()

        for inputs in train_loader:
            inputs = {k: inputs[k].to(device) for k in inputs}
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            cur_step += 1

            if cur_step % step_interval == 0:
                print(f"(step {cur_step}) Avg training loss: {total_loss/cur_step}")
                scheduler.step(total_loss-last_loss)
                last_loss = total_loss
            print(f"(step {cur_step}) Avg training loss: {total_loss/cur_step}")
            break
        # eval_nwp(model, test_loader)
        # print(f"(epoch {epoch}) Avg training loss: {total_loss/len(train_loader)}")

    # pure_name = model_name.replace('/', '_')
    # os.makedirs(os.path.join(path, pure_name), exist_ok=True)

    # for inputs in train_loader:
    #     torch.onnx.export(model, inputs,
    #         os.path.join(path, pure_name, f"{pure_name}.onnx"),
    #         export_params=True, verbose=0, training=1, opset_version=13,
    #         do_constant_folding=False, use_external_data_format=True,
    #         input_names=['input_ids'])
    #     break

with open("./nlp_nwp_zoo", 'r') as f:
    models = [s.split(':')[0] for s in f.readlines()]
    for model in models:
        print(model)
        train_nwp(model)

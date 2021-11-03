import torchtext
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch
import logging
import os
import pickle

os.environ["TOKENIZERS_PARALLELISM"] = "false"

wiki = torchtext.datasets.WikiText103(root='~/experiment', split='train')
max_text_length = 256
tokenizer = None

path = '/users/fanlai/experiment/nwp_zoo'

def collate(examples):
    if tokenizer._pad_token is None:
        return pad_sequence(examples, batch_first=True)
    return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

def tokenize_datset(tokenizer, data, block_size=256):
    data_iter = []
    for text in data:
        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            data_iter.append(torch.tensor(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i + block_size])))

    return data_iter

def train_nwp(model_name):
    global tokenizer
    pure_name = model_name.replace('/', '_')

    device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_config(config)
    #model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.max_length = max_text_length
    model.config.max_position_embeddings = max_text_length
    tokenizer.max_length = max_text_length


    train_dataset = tokenize_datset(tokenizer, torchtext.datasets.WikiText103(root='~/experiment', split='train'))
    #test_dataset = tokenize_datset(tokenizer, torchtext.datasets.WikiText103(root='~/experiment', split='train'))

    with open(f'~/experiment/{pure_name}_train', 'wb') as fout:
        pickle.dump(train_dataset, fout)

    # with open(f'~/experiment/{pure_name}_test', 'wb') as fout:
    #     pickle.dump(test_dataset, fout)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, verbose=True, min_lr=0, factor=0.5)

    EPOCHS = 100
    step_interval = 10000
    cur_step = 0

    model = model.to(device=device)

    for epoch in range(EPOCHS):
        total_loss = last_loss = 0

        for inputs in train_loader:
            inputs = inputs.to(device=device)
            optimizer.zero_grad()
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            cur_step += 1

            if cur_step % step_interval == 0:
                print(f"(step {cur_step}) Avg training loss: {total_loss/cur_step}")
                scheduler.step(total_loss-last_loss)
                last_loss = total_loss

            #print(f"loss {loss.item()}")
        #scheduler.step(total_loss/len(train_loader))
        print(f"(epoch {epoch}) Avg training loss: {total_loss/len(train_loader)}")

    # pure_name = model_name.replace('/', '_')
    # os.makedirs(os.path.join(path, pure_name), exist_ok=True)

    # for inputs in train_loader:
    #     torch.onnx.export(model, inputs,
    #         os.path.join(path, pure_name, f"{pure_name}.onnx"),
    #         export_params=True, verbose=0, training=1, opset_version=13,
    #         do_constant_folding=False, use_external_data_format=True,
    #         input_names=['input_ids'])
    #     break

train_nwp("bert-base-cased")


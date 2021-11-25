import torchtext
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForMaskedLM
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch
import logging
import os
import pickle

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def collate(examples, tokenizer):
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


def eval_nlp_nwp(model, test_loader, device=torch.device("cuda")):
    total_loss = 0
    model.eval()
    model = model.to(device=device)

    for inputs in test_loader:
        inputs = inputs.to(device=device)
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss

        total_loss += loss.item()

    print(f"Eval loss: {total_loss/len(test_loader)}")
    return 0, total_loss/len(test_loader)

def train_nlp_nwp(model, tokenizer, train_loader, optimizer, device=torch.device("cuda"), scheduler=None):
    total_loss = last_loss = cur_step = 0
    model.train()

    for inputs in train_loader:
        inputs = inputs.to(device=device)
        optimizer.zero_grad()
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        cur_step += 1

        if cur_step % 100 == 0:
            print(f"(step {cur_step}) Avg training loss: {total_loss/cur_step}")
            scheduler.step(total_loss-last_loss)
            last_loss = total_loss
        break

    print(f"Avg training loss: {total_loss/len(train_loader)}")

def load_nwp_tokenizer(name, max_text_length=256):
    # name = name.replace('/', '_')

    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.max_length = max_text_length
    return tokenizer

def load_nwp_model(name, max_text_length=256):
    # name = name.replace('/', '_')
    max_text_length = max(256, max_text_length)
    config = AutoConfig.from_pretrained(name)
    config.max_length = max_text_length
    config.max_position_embeddings = max_text_length
    model = AutoModelForMaskedLM.from_config(config)
    #model = AutoModelForCausalLM.from_pretrained(model_name)
    return model
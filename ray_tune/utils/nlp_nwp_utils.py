import logging
import os
import pickle

import torch
import torchtext
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForMaskedLM, AutoTokenizer)

#os.environ["TOKENIZERS_PARALLELISM"] = "false"
BLOCK_SIZE = 128

def tokenize_datset(tokenizer, data, block_size=BLOCK_SIZE):

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
    steps = 1e-3
    for inputs in test_loader:
        inputs = {k: inputs[k].to(device) for k in inputs}
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()
        steps += 1
    logging.info(f"Eval loss: {total_loss/steps}")
    return 0, total_loss/len(test_loader)

def train_nlp_nwp(model, tokenizer, train_loader, optimizer, device=torch.device("cuda"), scheduler=None):
    total_loss = last_loss = cur_step = 0
    model.train()
    eval_step = 1/20.
    breakdown_length = int(len(train_loader) * eval_step)

    for inputs in train_loader:
        inputs = {k: inputs[k].to(device) for k in inputs}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        cur_step += 1

        if cur_step == breakdown_length:
            break
        if scheduler is not None:
            scheduler.step()
        # if cur_step % 100 == 0:
        #     logging.info(f"(step {cur_step}) Avg training loss: {total_loss/cur_step}")
        #     scheduler.step(total_loss-last_loss)
        #     last_loss = total_loss

    logging.info(f"Avg training loss: {total_loss/cur_step}")

def load_nwp_tokenizer(name, max_text_length=BLOCK_SIZE):
    # name = name.replace('/', '_')

    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.max_length = max_text_length
    return tokenizer

def load_nwp_model(name, max_text_length=BLOCK_SIZE):
    max_text_length = max(BLOCK_SIZE, max_text_length)
    config = AutoConfig.from_pretrained(name)
    config.max_length = max_text_length
    config.max_position_embeddings = max_text_length
    model = AutoModelForMaskedLM.from_config(config)
    #model = AutoModelForCausalLM.from_pretrained(model_name)

    return model

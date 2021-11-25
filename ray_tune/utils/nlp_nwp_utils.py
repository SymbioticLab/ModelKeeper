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
    data_iter = []
    for text in data:
        # tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        tokenized_text = tokenizer.encode(text, max_length=512)
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            data_iter.append(torch.tensor(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i + block_size])))

    return data_iter


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

    print(f"Avg training loss: {total_loss/len(train_loader)}")

def load_nwp_model(name, max_text_length=256):
    # name = name.replace('/', '_')

    tokenizer = AutoTokenizer.from_pretrained(name)
    config = AutoConfig.from_pretrained(name)
    config.max_length = max_text_length
    config.max_position_embeddings = max_text_length
    model = AutoModelForMaskedLM.from_config(config)
    #model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.max_length = max_text_length
    return model, tokenizer
import inspect
import logging
import os
import pickle

import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, Trainer, TrainingArguments,
                          get_linear_schedule_with_warmup)

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                datefmt='%H:%M:%S',
                level=logging.INFO,
                handlers=[
                    logging.StreamHandler()
                ])

device = 'cuda'

def eval_nlp(model, test_loader):

    def compute_metrics(logits, labels):
        predictions = torch.argmax(logits, axis=-1)
        return torch.sum(predictions == labels)/len(labels)
        #return metric.compute(predictions=predictions, references=labels)

    model.eval()
    total_acc = total_loss = 0.
    for inputs in test_loader:
        inputs = {k: inputs[k].to(device=device) for k in inputs}
        outputs = model(**inputs)
        total_loss += outputs.loss.item()
        total_acc += compute_metrics(outputs.logits, inputs['labels'])

    logging.info(f"Eval loss: {total_loss/len(test_loader)}, accuracy: {total_acc*100./len(test_loader)}")


def train_nlp(model, tokenizer):

    train_dataset = load_dataset("imdb", split="train")
    test_dataset = load_dataset("imdb", split="test")
    train_dataset = train_dataset.rename_column('label', 'labels')
    test_dataset = test_dataset.rename_column('label', 'labels')

    train_dataset = train_dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, padding=True), batched=True)
    test_dataset = test_dataset.map(lambda batch: tokenizer(batch["text"], truncation=True, padding=True), batched=True)


    train_dataset.set_format(type='torch', columns=['attention_mask', 'input_ids', 'token_type_ids', 'labels'])
    test_dataset.set_format(type='torch', columns=['attention_mask', 'input_ids', 'token_type_ids', 'labels'])


    batch_size = 12
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    WARMUP_STEPS = int(0.2*len(train_loader))
    EPOCHS = 30
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS,
                                 num_training_steps=len(train_loader)*EPOCHS)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, verbose=True, min_lr=0, factor=0.2)

    model = model.to(device=device)
    for epoch in range(EPOCHS):
        total_loss = 0.
        for inputs in train_loader:
            inputs = {k: inputs[k].to(device=device) for k in inputs}
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss#[0]

            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            logging.info(f"loss {loss.item()}")
            scheduler.step()

        #scheduler.step(total_loss/len(train_loader))

        #if epoch % 5 == 0:
        eval_nlp(model, test_loader)

        logging.info(f"(epoch {epoch}) Avg training loss: {total_loss/len(train_loader)}")


def load_model(name):
    max_text_length = 256
    tokenizer = AutoTokenizer.from_pretrained(name, model_max_length=max_text_length)
    #model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
    config = AutoConfig.from_pretrained(name)
    config.num_labels = 2

    model = AutoModelForSequenceClassification.from_config(config)

    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.config.max_length = max_text_length

    return model, tokenizer

model, tokenizer = load_model("albert-base-v2")
train_nlp(model, tokenizer)


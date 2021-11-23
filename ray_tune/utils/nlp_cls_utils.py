from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

from datasets import load_dataset
from transformers import Trainer
import numpy as np
from datasets import load_metric
import pickle
import torch, os
import inspect
from transformers import TrainingArguments, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import logging

total_steps = 20

def eval_nlp_cls(model, test_loader, device=torch.device("cuda")):

    def compute_metrics(logits, labels):
        predictions = torch.argmax(logits, axis=-1)
        return torch.sum(predictions == labels)/len(labels)
        #return metric.compute(predictions=predictions, references=labels)

    model.eval()
    total_acc = total_loss = 0.
    step = 0
    for inputs in test_loader:
        inputs = {k: inputs[k].to(device=device) for k in inputs}
        outputs = model(**inputs)
        total_loss += outputs.loss.item()
        total_acc += compute_metrics(outputs.logits, inputs['labels'])

        step += 1
        if step > total_steps:
            break
    # logging.info(f"Eval loss: {total_loss/len(test_loader)}, accuracy: {total_acc*100./len(test_loader)}")
    return total_acc/len(test_loader), total_loss/len(test_loader)
    

def train_nlp_cls(model, tokenizer, train_loader, optimizer, device=torch.device("cuda"), scheduler=None):

    model = model.to(device=device)
    # for epoch in range(EPOCHS):
    total_loss = 0.
    step = 0
    for inputs in train_loader:
        inputs = {k: inputs[k].to(device=device) for k in inputs}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss#[0]

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        step += 1
        if step > total_steps:
            break
        #scheduler.step(total_loss/len(train_loader))
    model.to(device='cpu')


def load_cls_model(name, num_labels=5):
    max_text_length = 256
    tokenizer = AutoTokenizer.from_pretrained(name, model_max_length=max_text_length)
    config = AutoConfig.from_pretrained(name)
    config.num_labels = num_labels

    model = AutoModelForSequenceClassification.from_config(config)

    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.config.max_length = max_text_length

    return model, tokenizer
    

def main():
    device = 'cpu'
    model, tokenizer = load_cls_model("albert-base-v2")
    train_dataset = load_dataset("yelp_review_full", split="train")
    test_dataset = load_dataset("yelp_review_full", split="test")
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
    train_nlp_cls(model, tokenizer, train_loader, optimizer, device, scheduler)
    eval_nlp_cls(model, test_loader)

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from datasets import load_dataset
from transformers import Trainer
import numpy as np
from datasets import load_metric
import pickle
import torch, os
from fnmatch import fnmatch

path = '/mnt/transformers/'
padding_length = 256

def clean_transfomers():

    file_dir = '/users/fanlai/transformers/src/transformers/models'
    ans = []

    def extract_model_names(file):
        global ans 

        #print(file)
        with open(file) as fin:
            lines = fin.readlines()
        start = -1

        for i in range(len(lines)):
            if 'PRETRAINED_MODEL_ARCHIVE_LIST = [' in lines[i]:
                start = i 

            if ']' in lines[i] and start != -1:
                for x in lines[start+1:i]:
                    ans.append(x)
                start = -1

    pattern = "*.py"

    for path, subdirs, files in os.walk(file_dir):
        for name in files:
            if fnmatch(name, pattern):
                extract_model_names(os.path.join(path, name))


    ans = [x.strip() for x in ans if '# See all' not in x]
    dummy_ans = []

    for x in ans:
        try:
            dummy_ans.append(x.strip().split()[0].replace(',','').replace('"', ''))
        except Exception as e:
            print(f"===Failed {e}, {x}")

    dummy_ans = [x for x in list(set(dummy_ans)) if '#' not in x]

    blacklist = ['chinese', 'german', 'japanese', 'finnish', 'dutch', ]
    ans = []
    for model_path in dummy_ans:
        flag = False
        for b in blacklist:
            if b in model_path:
                flag = True
                break
        if not flag:
            ans.append(model_path)

    with open(f'results', 'w') as fout:
        for line in ans:
            fout.writelines(line+'\n')

    with open('transformers.pkl', 'wb') as fout:
        pickle.dump(ans, fout)
        
def validate_models():

    raw_datasets = load_dataset('imdb')['test'].select(range(1))
    metric = load_metric("accuracy")
    from transformers import TrainingArguments

    training_args = TrainingArguments("test_trainer")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def quick_test(model, raw_datasets, tokenizer, name, pad):

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)
            
        if pad:
            #inputs = tokenizer(raw_datasets['text'][1], return_tensors="pt")
            labels = torch.tensor([raw_datasets['label'][0]]).unsqueeze(0)  # Batch size 1

            inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            outputs = model(**inputs, labels=labels)
            print(outputs.loss)
        else:
            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=None,
                eval_dataset=tokenized_datasets,
                compute_metrics=compute_metrics,
            )

            print(trainer.evaluate())

    with open('/users/fanlai/transformers.pkl', 'rb')  as fin:
        model_names = pickle.load(fin)

    with open('/users/fanlai/success_seq_cls', 'w') as fout:
        for name in model_names:
            try_pad = False
            for i in range(2):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(name)
                    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)

                    if try_pad:
                        model.max_seq_length = padding_length
                        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                        
                    quick_test(model, raw_datasets, tokenizer, name, try_pad)

                    sum_params = sum([param.data.numel() for param in model.parameters()])
                    print(f"**** Model {name} success, params {sum_params} ****")
                    #print(sum_params)
                    fout.writelines(f"{name}: {sum_params}\n")
                    break
                except Exception as e:
                    if try_pad:
                        print(f"==== Model {name} fails because of {e} ====")
                    try_pad = True

    ans = set()
    unique_ans = []
    blacklist = ['chinese', 'german', 'japanese', 'finnish', 'dutch', ]

    with open(file) as fin:
        lines = fin.readlines()
        for line in lines:
            is_en = True
            for bl in blacklist:
                if bl in line:
                    is_en = False
                    break
            if is_en:
                params = int(line.strip().split()[1])
                if params not in ans:
                    ans.add(params)
                    unique_ans.append(line)

        unique_ans.sort()

    with open('success_seq_cls', 'w') as fout:
        for line in unique_ans:
            fout.writelines(line)

def dump_models(file):
    with open(file) as fin:
        lines = fin.readlines()
        model_names = [x.split(':')[0] for x in lines]

    for name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)

        pure_name = name.replace('/', '_')
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.config.max_length = padding_length

        try:
            inputs = tokenizer("Hello, my dog is cute", return_tensors='pt')#, padding="max_length", max_length=padding_length)
            
            input_names = sorted(list(inputs.keys()))
            dummy_inputs = tuple([inputs[val] for val in input_names])

            try:
                os.mkdir(os.path.join(path, pure_name))
            except Exception as e:
                pass

            torch.onnx.export(model, dummy_inputs, 
                os.path.join(path, pure_name, f"{pure_name}.onnx"),
                export_params=True, verbose=0, training=1, opset_version=13,
                do_constant_folding=True, use_external_data_format=True,
                input_names=input_names)

            print(f"**** Model {pure_name}  successes ****")
        except Exception as e:
            print(f"==== Model {pure_name} fails, {e} ====")

dump_models("new_success_seq_cls")

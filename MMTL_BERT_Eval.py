# Confirm we can import from metal
import sys
sys.path.append('../../metal')
# Import custom metal package with https://stackoverflow.com/questions/30292039/pip-install-forked-github-repo
import metal
import os
import pickle
import dill
from metal.mmtl.trainer import MultitaskTrainer
from metal.mmtl.payload import Payload
from metal.mmtl import MetalModel
from metal.mmtl.task import ClassificationTask
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert import BertTokenizer, BertModel

# Import other dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import pandas as pd
import csv
import sys
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--max_seq_length', default="150", help='Max size of the input')
parser.add_argument('--batch_size', default="32", help='Batch size of training')
parser.add_argument('--dailydialogue_data_path', default="./data", help='The file path of the dailydialogue dataset')
parser.add_argument('--emotionlines_data_path', default="./data", help='The file path of the emotionlines dataset')
parser.add_argument('--results_output', default="./results", help='The file path of the training results and output model')
parser.add_argument('--model_name', default="mmtl_BERT_model", help='Name of the saved model')
parser.add_argument('--num_epochs', default="1", help='How many epochs to run training for')
args = parser.parse_args()
MAX_SEQ_LENGTH = int(args.max_seq_length)
BATCH_SIZE = int(args.batch_size)
DAILYDIALOGUE_DATA_PATH = args.dailydialogue_data_path
EMOTIONLINES_DATA_PATH = args.emotionlines_data_path
RESULTS_OUTPUT = args.results_output
MODEL_NAME = args.model_name
NUM_EPOCHS = int(args.num_epochs)

if not os.path.exists(RESULTS_OUTPUT):
    os.mkdir(RESULTS_OUTPUT)

def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar, error_bad_lines=False)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines

def create_full_BERT_input(folder_path, text_column_name, label_column_name, context_column_name=None):
    
    label_map = {}
        
    def create_BERT_inputs_from_file(file_name, text_column_name, label_column_name, context_column_name):

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        raw_tsv_list = read_tsv(file_name)
        headers = raw_tsv_list.pop(0)
        
        def get_BERT_input_for_example(datum):

            if not datum[headers.index(label_column_name)] in list(label_map.keys()):
                # We need to add one to this as MeTaL does not accept 0 as a valid label
                label_map[datum[headers.index(label_column_name)]] = len(list(label_map.values())) + 1

            # We tokenize our initial data
            tokens_a = tokenizer.tokenize(datum[headers.index(text_column_name)])
            if context_column_name == None:
                tokens_b = []
            else:
                tokens_b = tokenizer.tokenize(datum[headers.index(context_column_name)])
            label = label_map[datum[headers.index(label_column_name)]]

            # We create our input token string by adding BERT's necessary classification and separation tokens
            # We also create the segment id list which is 0 for the first sentence, and 1 for the second sentence
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            
            if len(tokens_b) > 0:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            if len(input_ids) < MAX_SEQ_LENGTH:
                # Zero-pad up to the sequence length.
                padding = [0] * (MAX_SEQ_LENGTH - len(input_ids))

                input_ids += padding
                segment_ids += padding
            else:
                input_ids = input_ids[0:MAX_SEQ_LENGTH]
                segment_ids = segment_ids[0:MAX_SEQ_LENGTH]

            return (input_ids, segment_ids, label)
        
        tokenized_inputs = list(map(get_BERT_input_for_example, raw_tsv_list))

        all_input_ids = torch.tensor([tokenized_input[0] for tokenized_input in tokenized_inputs], dtype=torch.long)
        all_segment_ids = torch.tensor([tokenized_input[1] for tokenized_input in tokenized_inputs], dtype=torch.long)
        all_label_ids = torch.tensor([tokenized_input[2] for tokenized_input in tokenized_inputs], dtype=torch.long)

        return (all_input_ids, all_segment_ids, all_label_ids, label_map)
    
    train_path = folder_path + "/train.tsv"
    dev_path = folder_path + "/dev.tsv"
    test_path = folder_path + "/test.tsv"
    
    train_inputs = create_BERT_inputs_from_file(train_path, text_column_name, label_column_name, context_column_name)
    dev_inputs = create_BERT_inputs_from_file(dev_path, text_column_name, label_column_name, context_column_name)
    test_inputs = create_BERT_inputs_from_file(test_path, text_column_name, label_column_name, context_column_name)
    
    return (train_inputs, dev_inputs, test_inputs)

class InputTask:
    train_inputs = ()
    dev_inputs = ()
    test_inputs = ()
    no_of_lables = 0
    batch_size = 0
    
    def __init__(self, path, batch_size, text_column_name, label_column_name, context_column_name=None):
        self.path = path
        self.batch_size = batch_size
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name
        self.context_column_name = context_column_name


database_tasks = {
    "dailydialoguecontext": InputTask(
        path = DAILYDIALOGUE_DATA_PATH,
        batch_size = BATCH_SIZE,
        text_column_name = "dialogue",
        label_column_name = "emotion",
        context_column_name = "context",
    ),
    "friendscontext": InputTask(
        path = EMOTIONLINES_DATA_PATH,
        batch_size = BATCH_SIZE,
        text_column_name = "utterance",
        label_column_name = "emotion",
        context_column_name = "context",
    ),
}

for task_name in database_tasks:
    database_folder_path = database_tasks[task_name].path
    text_column_name = database_tasks[task_name].text_column_name
    label_column_name = database_tasks[task_name].label_column_name
    context_column_name = database_tasks[task_name].context_column_name
    
    train_inputs, dev_inputs, test_inputs = create_full_BERT_input(database_folder_path, text_column_name, label_column_name, context_column_name)
    
    database_tasks[task_name].train_inputs = train_inputs
    database_tasks[task_name].dev_inputs = dev_inputs
    database_tasks[task_name].test_inputs = test_inputs
    
    database_tasks[task_name].no_of_lables = len(train_inputs[3])

input_module = BertModel.from_pretrained('bert-base-uncased')
input_module.config.max_position_embeddings = MAX_SEQ_LENGTH

tasks = []

for task_name in database_tasks:
    classification_task = ClassificationTask(
        name=task_name, 
        input_module=input_module, 
        head_module= torch.nn.Linear(768, database_tasks[task_name].no_of_lables))
    
    tasks.append(classification_task)

for i, x in enumerate(database_tasks):
    print(i)
    print(x)

model = MetalModel(tasks, verbose=True)

def create_BERT_tensor(input_ids, segment_ids):
    return(torch.cat((input_ids, segment_ids), 1))

payloads = []
splits = ["train", "valid", "test"]
for i, database_task_name in enumerate(database_tasks):
    input_task = database_tasks[database_task_name]
    
    payload_train_name = f"Payload{i}_train"
    payload_dev_name = f"Payload{i}_dev"
    payload_test_name = f"Payload{i}_test"
    task_name = database_task_name
    
    batch_size = input_task.batch_size
    
    train_inputs = input_task.train_inputs
    dev_inputs = input_task.dev_inputs
    test_inputs = input_task.test_inputs
    
    train_X = {"data": create_BERT_tensor(train_inputs[0], train_inputs[1])}
    dev_X = {"data": create_BERT_tensor(dev_inputs[0], dev_inputs[1])}
    test_X = {"data": create_BERT_tensor(test_inputs[0], test_inputs[1])}
    
    train_Y = train_inputs[2]
    dev_Y = dev_inputs[2]
    test_Y = test_inputs[2]
    
    payload_train = Payload.from_tensors(payload_train_name, train_X, train_Y, task_name, "train", batch_size=batch_size)
    payload_dev = Payload.from_tensors(payload_dev_name, dev_X, dev_Y, task_name, "valid", batch_size=batch_size)
    payload_test = Payload.from_tensors(payload_test_name, test_X, test_Y, task_name, "test", batch_size=batch_size)
    
    payloads.append(payload_train)
    payloads.append(payload_dev)
    payloads.append(payload_test)

model = MetalModel(tasks, verbose=False)

os.environ['METALHOME'] = RESULTS_OUTPUT

trainer = MultitaskTrainer()
trainer.config["checkpoint_config"]["checkpoint_every"] = 1
trainer.config["checkpoint_config"]["checkpoint_best"] = True

for i in range(NUM_EPOCHS):
    scores = trainer.train_model(
        model, 
        payloads, 
        n_epochs=1, 
        log_every=1,
        progress_bar=False,
    )
    print("Epoch number {0} scores:".format(i))
    print(scores)

filename = OUTPUT_PATH + '/'+MODEL_NAME+'.out'

with open(filename, 'wb') as pickle_file:
    torch.save(model.state_dict(), pickle_file)

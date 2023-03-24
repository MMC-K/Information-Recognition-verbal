from __future__ import print_function, division
import os
import glob
import json
import copy
import pickle

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import re
import collections
from torch._six import string_classes
np_str_obj_array_pattern = re.compile(r'[SaUO]')

SENTIMENT_CLASSES = [
    2,
    1,
    0
]

class EmotionBaseDataset(Dataset):
    def __init__(
        self,
        data_path,
        sentiment_classes=SENTIMENT_CLASSES
    ):
        self.data_path = data_path
        self.sentiment_classes = sentiment_classes
    
    def _load_json(self, data_path):
        with open(data_path, "r") as f:
            return json.load(f)

def default_history_formatting(dialogue_history, person_id, is_reverse=True, sep=' '):
    dialogue = [utterance['text'] for utterance in dialogue_history]
    if is_reverse:
        dialogue.reverse()
    return sep.join(dialogue)

def history_formatting_with_pid(dialogue_history, person_id, is_reverse=True, sep=' '):
    dialogue = [ "{}: {}".format(utterance['person_id'], utterance['text']) for utterance in dialogue_history]
    if is_reverse:
        dialogue.reverse()
    return sep.join(dialogue)

def self_utterance_history(dialogue_history, person_id, is_reverse=True, sep=' '):
    dialogue = [utterance['text'] for utterance in dialogue_history if utterance['person_id']==person_id]
    if is_reverse:
        dialogue.reverse()
    return sep.join(dialogue)

def buddy_utterance_history(dialogue_history, person_id, is_reverse=True, sep=' '):
    dialogue = [utterance['text'] for utterance in dialogue_history if utterance['person_id']!=person_id]
    if is_reverse:
        dialogue.reverse()
    return sep.join(dialogue)

def utterance_dialoge_history_formatting(utterance, dialogue_history):
    return "utterance: {} history: {}".format(utterance, dialogue_history)

def get_history_proc(proc_name):
    if 'default':
        return default_history_formatting
    elif 'self':
        return self_utterance_history
    elif 'buddy':
        return buddy_utterance_history

def get_input_proc(proc_name):
    return utterance_dialoge_history_formatting

class EmotionExampleDataset(EmotionBaseDataset):
    def __init__(
        self,
        data_path,
        sentiment_classes=SENTIMENT_CLASSES
    ):
        super(EmotionExampleDataset, self).__init__(data_path, sentiment_classes)

        self.examples = self._create_example(data_path)

    def _create_example(self, data_path):
        data = pd.read_csv(data_path)
        examples = []
        data.dropna(axis='index', inplace=True)

        for q, label in zip(data['sentence'], data['label']):
            examples.append({
                'text':q,
                'label':label
            })
        
        return examples

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        example = self.examples[idx]

        input_txt = example['text']

        return {
            'text': input_txt,
            'label': torch.tensor(np.array(example['label']), dtype=torch.long)
        }


collate_variable_length_err_msg_format = (
    "collate_variable_length: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def collate_variable_length(batch):
    elem = batch[0]

    if isinstance(elem, collections.abc.Mapping):
        return {key: _collate_variable_length([d[key] for d in batch]) for key in elem}
    else:
        raise RuntimeError('each element must be the Dictionary types')


def _collate_variable_length(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""


    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(collate_variable_length_err_msg_format.format(elem.dtype))

            return _collate_variable_length([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: [d[key] for d in batch] for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(_collate_variable_length(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        return [_collate_variable_length(elem) for elem in it]

    raise TypeError(collate_variable_length_err_msg_format.format(elem_type))


if __name__ == "__main__":
    data_path = "/home/jonghwi/hdd/hdd/data/복합대화_감성분석_데이터세트/문장 단위 감성 말뭉치/모델 학습 및 평가 데이터(최종)/train_data.csv"
    dataset = EmotionExampleDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=16,
                        shuffle=True, num_workers=0)
    
    print(len(dataset))
    for idx, data in enumerate(dataloader):
        print(data)
        if idx > 1:
            exit()

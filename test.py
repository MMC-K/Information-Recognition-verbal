from __future__ import print_function, division
import os
import sys
import json
import time
import shutil
import argparse
from numpy.lib.arraypad import pad

from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from transformers import AutoTokenizer
from transformers import Adafactor

from data_utils import EmotionExampleDataset, SENTIMENT_CLASSES, collate_variable_length
from models import get_model


def get_model_arch_pretrained(config_path):
    with open(config_path, "r") as f:
        in_json = json.load(f)
        return in_json['architectures'][0], in_json['_name_or_path']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                        default='./datasets/test_samples.csv',
                        type=str, help="path to the training data")

    parser.add_argument("--history_proc",
                        default='default',
                        help="type of history processing")

    parser.add_argument("--batch_size",
                        default=16,
                        type=int, help="mini batch size")

    parser.add_argument("--hf_path",
                        default='./model/T5-mean-m_KETI-AIR_ke-t5-base_default/weights/',
                        help="path to hf model")
    
    parser.add_argument("--model_type",
                        default='T5-mean-m',
                        type=str, help="type of model")
    args = parser.parse_args()

    pretrained_model_path = 'KETI-AIR/ke-t5-base'
    checkpoint = torch.load('./model/T5-mean-m_KETI-AIR_ke-t5-base_default/weights/best.pth')
    model = get_model(args.model_type, pretrained_model_path, SENTIMENT_CLASSES)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    dataset = EmotionExampleDataset(args.data_path, sentiment_classes=SENTIMENT_CLASSES)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=0,
                                  collate_fn=collate_variable_length)
    
    model.eval()

    sentences = []
    preds = []
    labels = []

    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), desc='proc...'):
            inputs = tokenizer(batch['text'], padding=True,
                            truncation='longest_first', return_tensors='pt')

            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            input_length = input_ids.size()[-1]
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

            if args.model_type == 'seq2seq':
                label = tokenizer(batch['label_txt'], padding='max_length',
                                truncation=True, return_tensors='pt').input_ids.cuda()
            else:
                label = batch['label'].cuda()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels = label
            )

            logits = outputs[1]

            if args.model_type == 'seq2seq':
                _, predicted = torch.max(logits, -1)
                predicted = predicted.cpu().numpy()
                predicted = [tokenizer.decode(x) for x in predicted]

            else:
                _, predicted = torch.max(logits, 1)
                predicted = predicted.cpu().numpy()
                
            label = label.cpu().numpy()
            preds.extend(predicted)
            labels.extend(label)
            sentences.extend(batch['text'])

    error_analysis = pd.DataFrame({'sentence':sentences, 'label':labels, 'prediction':preds})
    print(error_analysis)

if __name__ == "__main__":
    main()



# python eval_model.py --data_path ./datasets/복합대화_감성분석_데이터세트/문장 단위 감성 말뭉치/모델 학습 및 평가 데이터(최종)/test_data.csv --batch_size 16  --model_type T5-mean-m
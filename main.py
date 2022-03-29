import os
from transformers import AutoTokenizer, GPT2LMHeadModel
from tqdm import tqdm

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re

from random import sample

from model import GPT2FewshotClassifier
from preprocess import prepare_data
from metrics import accuracy

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# def build_prompt_text(sent):
#     return "문장: " + sent + ' 감정: '


def build_prompt_text(sent):
    return '감정 분석 문장: ' + sent + ' 결과: '


def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", sent)
    return sent_clean


def main(train_fewshot_samples, test_data, model):
    real_labels = []
    pred_tokens = []

    for i, (test_sent, test_label) in enumerate(tqdm(test_data[['document', 'label']].values)):
        tokens = tokenizer('<s>')['input_ids']

        for ex in train_fewshot_samples[i]:
            example_text, example_label = ex
            cleaned_example_text = clean_text(example_text)
            appended_prompt_example_text = build_prompt_text(cleaned_example_text)
            appended_prompt_example_text += '긍정' if example_label == 1 else '부정'

            # tokens += vocab[tokenizer(appended_prompt_example_text)]
            tokens += tokenizer(appended_prompt_example_text)['input_ids']

        cleaned_sent = clean_text(test_sent)
        appended_prompt_sent = build_prompt_text(cleaned_sent)
        test_tokens = tokenizer(appended_prompt_sent)['input_ids']
        
        tokens += test_tokens
        
        model.eval()
        with torch.no_grad():
            torch_pred = torch.argmax(model(torch.tensor([tokens]).cuda()), axis=-1)
        # torch_pred = torch_model.gpt2.generate

        pos = tokenizer('긍정')['input_ids']
        negative = tokenizer('부정')['input_ids']            
        torch_label = pos if test_label == 1 else negative
      
        pred_tokens.append(torch_pred[0])
        real_labels.append(torch_label[0])
        
    accuracy(pred_tokens, real_labels)


if __name__ == "__main__":
    tr_path = 'data/ratings_train.csv'
    tst_path = 'data/ratings_test.csv'
    checkpoints = 'skt/ko-gpt-trinity-1.2B-v0.5'
    # checkpoints = 'skt/kogpt2-base-v2'
    tokenizer = AutoTokenizer.from_pretrained(checkpoints)
    model = GPT2FewshotClassifier(checkpoints).cuda()
    test_data, train_fewshot_samples = prepare_data(tr_path, tst_path, tokenizer, sample_size=5000)
    
    main(train_fewshot_samples, test_data, model)
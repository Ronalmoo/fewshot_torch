import random
import pandas as pd
from random import sample


# tr_path = 'data/ratings_train.csv'
# tst_path = 'data/ratings_test.csv'



def prepare_data(tr_path, tst_path, tokenizer, sample_size=5000):
    train_data = pd.read_csv(tr_path).dropna()
    test_data = pd.read_csv(tst_path).dropna()
    train_fewshot_data = []
    for train_sent, train_label in train_data[['document', 'label']].values:
        # tokens = vocab[tokenizer(train_sent)]
        tokens = tokenizer(train_sent)['input_ids']

        if len(tokens) <= 25:
            train_fewshot_data.append((train_sent, train_label))

    train_fewshot_samples = []

    for _ in range(sample_size):
        fewshot_examples = sample(train_fewshot_data, 30)
        train_fewshot_samples.append(fewshot_examples)

    if sample_size < len(test_data['id']):
        test_data = test_data.sample(sample_size, random_state=42)
    
    return test_data, train_fewshot_samples
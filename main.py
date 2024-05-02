import json
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from utils.loader import llm, emb_fn, strings
from utils.utils import prepare_data, set_all_seeds, save_predictions
from utils.optimizer import Protegi


N_EPOCHS = 1
set_all_seeds(42)


eval_dataset = pd.read_csv('./data/val_spam.csv')
train_dataset = pd.read_csv('./data/train_spam.csv')
beam_size = 3
cached_scores = {}
protegi = Protegi()
prompts = [strings['templates']['prediction_template']]
candidates = prompts
for epoch in range(1, N_EPOCHS+1):
    print("STARTING EPOCH", epoch)
    start = time.time()
    candidates = protegi.expand_candidates(candidates, train_dataset)
    for candidate in candidates:
        print(candidate)
        print('-------------------------')
    scores = protegi.score_candidates(candidates, eval_dataset, 'spam')
    for prompt in candidates:
        cached_scores[prompt] = scores[prompt]
    candidates = list(scores.keys())[:beam_size]

with open ('./artifacts/results/spam_results.json', 'w') as fp:
    json.dump(cached_scores,fp)



import json
import pandas as pd
import time
from utils.loader import strings
from utils.optimizer import Protegi

task = 'spam'
assert task in ['spam'], "this task is not supported"
N_EPOCHS = 5
salt = "020524"

eval_dataset = pd.read_csv(f'./data/val_{task}.csv')
train_dataset = pd.read_csv(f'./data/train_{task}.csv')
beam_size = 3
cached_scores = {}
protegi = Protegi(task=task, salt=salt)
prompts = [strings['templates'][task]['prediction_template']]
candidates = prompts

for epoch in range(1, N_EPOCHS+1):
    print("STARTING EPOCH", epoch)
    start = time.time()
    candidates = protegi.expand_candidates(candidates, train_dataset)
    scores = protegi.score_candidates(candidates, eval_dataset, 'spam')
    for prompt in candidates:
        cached_scores[prompt] = scores[prompt]
    with open (f'./artifacts/results/spam_results_{salt}.json', 'w') as fp:
        json.dump(cached_scores,fp)
    candidates = list(scores.keys())[:beam_size]
    for candidate in candidates:
        print(candidate)
        print('-----------------------------------------')



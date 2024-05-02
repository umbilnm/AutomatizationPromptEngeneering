import math
import numpy as np
import random
from tqdm import tqdm
import concurrent.futures
import requests
import urllib3

class BruteForceEvaluator:
    """ Brute Force Evaluator """
    def __init__(self, config):
        self.config = config

    def __call__(self, prompts, exs, task, predictor, scorer,
rounds=40, num_prompts_per_round=10, c=2.0, samples_per_eval=5, max_threads=1, verbose=True):
        sample_size = min(len(exs), int(self.config['eval_budget'] / len(prompts)))
        eval_exs = random.sample(exs, sample_size)

        while True:
            try:
                scores = scorer(predictor, prompts, eval_exs, max_threads=max_threads)
                break
            except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError, urllib3.exceptions.MaxRetryError):
                pass
        return scores

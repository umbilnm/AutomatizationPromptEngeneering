import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from utils.loader import llm, emb_fn, strings
from utils.utils import prepare_data, set_all_seeds, save_predictions
from utils.optimizer import Protegi

BATCH_SIZE = 16
N_EPOCHS = 10
set_all_seeds(42)

prepare_data('spam', '/home/umbilnm/python_ml/AutomatizationPromptEngeneering/data/spam.csv', [100, 1000]) 
df = pd.read_csv('/home/umbilnm/python_ml/AutomatizationPromptEngeneering/data/spam_100.csv').drop(
    columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4']
)
messages = df['v2']

protegi = Protegi()
prompts = [strings['templates']['prediction_template']]
for epoch in range(1, N_EPOCHS+1):
    print("STARTING EPOCH", epoch)
    start = time.time()

    candidates = protegi.expand_candidates(candidates)
    





import json 
import pandas as pd
import numpy as np
import os
import random
from utils.loader import strings 
from typing import List
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from langchain.prompts import PromptTemplate

def set_all_seeds(seed:int) -> None:
    # python's seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def prepare_data(dataset_name:str, dataset_path:str, sizes: List[int]) -> None: 
    df = pd.read_csv(dataset_path, encoding='cp1251')
    for size in sizes:
        df_sample = df.sample(n=size, replace=False, random_state=42)
        df_sample.to_csv(f'/home/umbilnm/python_ml/AutomatizationPromptEngeneering/data/{dataset_name}_{size}.csv', index=False)



def save_predictions(df:pd.DataFrame, text_col:str, pred_col:str, name:str) -> None:
    df[[text_col,pred_col]].to_csv(f'/home/umbilnm/python_ml/AutomatizationPromptEngeneering/data/predistions/{name}', index=False)


def calculate_metrics(labels: np.ndarray, preds: np.ndarray, task: str) -> dict:
    with open(f'./artifacts/json_mappings/{task}_map.json') as f:
        mapping = json.load(f)
    labels = np.vectorize(mapping.get)(labels)
    preds  = np.vectorize(mapping.get)(preds)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'recall':recall, 'precision':precision}
    
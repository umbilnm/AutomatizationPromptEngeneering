import pandas as pd
import numpy as np
import os
import random
from utils.loader import strings 
from typing import List
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


def combine_gradient_prompt(df: pd.DataFrame, true_col:str, pred_col:str, text_col:str) -> str:
    prediction_template = strings['templates']['prediction_template']
    indexes = np.where(df[true_col] != df[pred_col])[0]
    mapping = {1:'spam', 0:'ham'}
    mistakes_template = '''There are examples where models answer is incorrect:\n'''
    gradient_template = PromptTemplate(template=strings['templates']['gradient_template'],
                                       input_variables=['prompt', 'error_string', 'num_feedbacks'])   
    for idx in indexes:
        pred, true, message = df.iloc[idx, :][pred_col], df.iloc[idx][true_col], df.iloc[idx][text_col]
        mistakes_template += message + '\n'
        mistakes_template +=f'True: {mapping[true]}\nPredicted: {mapping[pred]}\n'
        mistakes_template += '------------------------------\n' 

    return gradient_template.format(prompt=prediction_template, error_string=mistakes_template, num_feedbacks='3')

def save_predictions(df:pd.DataFrame, text_col:str, pred_col:str, name:str) -> None:
    df[[text_col,pred_col]].to_csv(f'/home/umbilnm/python_ml/AutomatizationPromptEngeneering/data/predistions/{name}', index=False)
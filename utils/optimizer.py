from collections import OrderedDict
import json
from tqdm import tqdm
from typing import List
import logging
import pandas as pd
import numpy as np
from langchain.prompts import PromptTemplate
from utils.loader import strings, llm
from utils.utils import calculate_metrics

class Protegi:
    def __init__(self, task: str, salt:str):
        self.task = task
        self.salt = salt
        self.scores = {}
    def evaluate_batch(self, df:pd.DataFrame, prompt:str, pred_col:str, true_col:str, text_col:str) -> pd.DataFrame:
        messages = df[text_col]
        preds = []
        for message in tqdm(messages):
            prediction = llm.invoke(prompt.format(message=message)).content
            preds.append(prediction)
        df = df.copy()
        df[pred_col] = preds
        if self.task =='spam':

            df[pred_col] = df[pred_col].apply(lambda x: 'spam' if x=='Yes' else 'ham')

        return df[text_col].values, df[true_col].values, df[pred_col].values
    
    def _sample_errors_string(self, true_val:np.array, pred_val:np.array, text_val:np.array)-> str:
        indexes = np.where(true_val != pred_val)[0]
        mistakes_template = ''
        for idx in indexes:
            mistakes_template += text_val[idx] + '\n'
            mistakes_template +=f'True: {true_val[idx]}\nPredicted: {pred_val[idx]}\n'
            mistakes_template += '------------------------------\n' 

        return mistakes_template
    
    def parse_tagged_text(self, text:str, start_tag:str, end_tag:str):
        """ Parse text that is tagged with start and end tags."""
        texts = []
        while True:
            start_index = text.find(start_tag)
            if start_index == -1:
                break
            end_index = text.find(end_tag, start_index)
            if end_index == -1:
                break
            start_index += len(start_tag)
            texts.append(text[start_index:end_index].strip())
            text = text[end_index+len(end_tag):]
        return texts

    def _get_gradient(self, mistakes_template) -> str:
        prediction_template = strings['templates'][self.task]['prediction_template']
        gradient_template = PromptTemplate(template=strings['templates'][self.task]['gradient_template'],
                                        input_variables=['prompt', 'error_string', 'num_feedbacks']) 

        answer = llm.invoke(gradient_template.format(prompt=prediction_template, error_string=mistakes_template, num_feedbacks='3'))
        answer = self.parse_tagged_text(answer.content, '<START>', '<END>')
        res = [(t, mistakes_template) for t in answer]
        return res
    
    def generate_synonyms(self, prompt_section):
        """ Generate synonyms for a prompt section."""
        rewriter_prompt = f"Generate a variation of the following instruction while keeping the semantic meaning.\n\nInput: {prompt_section}\n\nOutput:"
        new_instructions = [llm.invoke(rewriter_prompt).content]
        return new_instructions
    
    
    def apply_gradient(self, prompt:str, error_str:str, feedback_str:str, steps_per_gradient:int):
        """ Incorporate feedback gradient into a prompt."""
        transformation_prompt = strings['templates'][self.task]['transformation_template']  
        res = llm.invoke(transformation_prompt.format(prompt=prompt, \
                                               error_str=error_str, feedback_str=feedback_str, steps_per_gradient=steps_per_gradient))
        new_prompts = []   
        new_prompts += self.parse_tagged_text(res.content, "<START>", "<END>")
        return new_prompts


    def expand_candidates(self, prompts:List[str], df:pd.DataFrame) -> List[str]:
        """ Expand a list of prompts by generating gradient-based successors and 
            synonyms for each section.
        """
 
        new_prompts = []
        for prompt in prompts:
            logging.info('Evaluating batch...')
            texts, labels, preds = self.evaluate_batch(df=df, prompt=prompt, pred_col='predictions',
                                                       true_col='v1', text_col='v2')
            
            new_task_sections = []
            
            logging.info('Sampling errors string')
            error_str = self._sample_errors_string(true_val=labels, pred_val=preds, text_val=texts)
            
            logging.info('Get gradient prompt')
            gradients = self._get_gradient(error_str)
            new_task_sections = []
            
            logging.info('Applying gradients')
            for feedback, error_string in gradients:
                tmp = self.apply_gradient(prompt, error_string, feedback, 3)
                new_task_sections += tmp

            mc_sampled_task_sections = []
            logging.info('Combine')
            new_sections = new_task_sections + mc_sampled_task_sections
            new_sections = list(set(new_sections)) # dedup
            tmp_new_prompts = [
                tmp 
                for tmp in new_sections
            ]
            new_prompts += tmp_new_prompts

        new_prompts += prompts # add originals
        new_prompts = list(set(new_prompts)) # dedup
        new_prompts = list(filter(lambda x: '{message}' in x, new_prompts))
        return new_prompts

    def score_candidates(self, candidates:List[str], eval_dataset:pd.DataFrame, task:str) -> dict:
        """ Score a list of candidates."""
        
        for candidate in candidates:
            _, labels, preds = self.evaluate_batch(df=eval_dataset, prompt=candidate, pred_col='predictions',
                                            true_col='v1', text_col='v2')
            metrics = calculate_metrics(labels, preds, task)
            self.scores[candidate] = metrics['f1']
            with open (f'./artifacts/results/{self.task}_results_{self.salt}.json', 'w') as fp:
                json.dump(self.scores,fp)
        return self.scores

from langchain.prompts import PromptTemplate
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from utils.loader import strings, llm 
from typing import List
import logging

class Protegi:
    def evaluate_batch(self, df:pd.DataFrame, prompt:str, pred_col:str, true_col:str, text_col:str) -> pd.DataFrame:
        messages = df[text_col]
        preds = []
        for message in messages:
            prediction = llm.invoke(prompt.format(message=message)).content
            preds.append(prediction)
        
        df[pred_col] = preds
        df[pred_col] = df[pred_col].apply(lambda x: 'spam' if x=='Yes' else 'ham')

        return df[text_col].values, df[true_col].values, df[pred_col].values
    
    def _sample_errors_string(self, true_val:np.array, pred_val:np.array, text_val:np.array)-> str:
        indexes = np.where(true_val != pred_val)[0]
        mapping = {0:'ham', 1:'spam'}
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
        prediction_template = strings['templates']['prediction_template']
        gradient_template = PromptTemplate(template=strings['templates']['gradient_template'],
                                        input_variables=['prompt', 'error_string', 'num_feedbacks']) 
        # mistakes_template = self._sample_errors_string(df, true_col, pred_col, text_col)  
        
        answer = llm.invoke(gradient_template.format(prompt=prediction_template, error_string=mistakes_template, num_feedbacks='3'))
        answer = self.parse_tagged_text(answer.content, '<START>', '<END>')
        res = [(t, mistakes_template) for t in answer]
        return res
    
    def generate_synonyms(self, prompt_section):
        """ Generate synonyms for a prompt section."""
        rewriter_prompt = f"Generate a variation of the following instruction while keeping the semantic meaning.\n\nInput: {prompt_section}\n\nOutput:"
        new_instructions = [llm.invoke(rewriter_prompt).content]
        # new_instructions = [x.content for x in new_instructions if x]
        return new_instructions
    
    
    def apply_gradient(self, prompt:str, error_str:str, feedback_str:str, steps_per_gradient:int):
        """ Incorporate feedback gradient into a prompt."""
        transformation_prompt = strings['templates']['transformation_template']  
        res = llm.invoke(transformation_prompt.format(prompt=prompt, \
                                               error_str=error_str, feedback_str=feedback_str, steps_per_gradient=steps_per_gradient))
        new_prompts = []   
        new_prompts += self.parse_tagged_text(res.content, "<START>", "<END>")
        return new_prompts


    def expand_candidates(self, prompts:List[str], df:pd.DataFrame, batch_size:int=10):
        """ Expand a list of prompts by generating gradient-based successors and 
            synonyms for each section.
        """
        minibatch = df.sample(batch_size, replace=False)

        new_prompts = []
        for prompt in prompts:
            logging.info('Evaluating batch...')
            # evaluate prompt on minibatch
            texts, labels, preds = self.evaluate_batch(df=minibatch, prompt=prompt, pred_col='predictions',
                                                       true_col='v1', text_col='v2')
            
            # get gradients
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

            logging.info('Generating synonyms')
            # generate synonyms
            mc_sampled_task_sections = []
            for sect in new_task_sections:
                mc_sects = self.generate_synonyms(sect)
                mc_sampled_task_sections += mc_sects
            # return mc_sampled_task_sections, new_task_sections
            logging.info('Combine')
            # combine
            new_sections = new_task_sections + mc_sampled_task_sections
            new_sections = list(set(new_sections)) # dedup
            tmp_new_prompts = [
                tmp 
                for tmp in new_sections
            ]
            
            # # filter a little
            # if len(new_sections) > self.opt['max_expansion_factor']:
            #     if self.opt['reject_on_errors']:
            #         error_exs = []
            #         for i, (t, l, p) in enumerate(zip(texts, labels, preds)):
            #             if l != p:
            #                 error_exs.append({'text': t, 'label': l})
            #         error_exs = random.sample(error_exs, min(len(error_exs), 16))

            #         # speed up a little
            #         tmp_new_prompts = random.sample(tmp_new_prompts, min(len(tmp_new_prompts), self.opt['max_expansion_factor'] * 2))

            #         error_scores = self.bf_eval(tmp_new_prompts, error_exs, task, gpt4, self.scorer, max_threads=self.max_threads)
            #         tmp_new_prompts = [tmp_new_prompts[i] for i in np.argsort(error_scores)[-self.opt['max_expansion_factor']:]]
            #     else:
            #         tmp_new_prompts = random.sample(tmp_new_prompts, 
            #             k=self.opt['max_expansion_factor'])

            new_prompts += tmp_new_prompts

        new_prompts += prompts # add originals
        new_prompts = list(set(new_prompts)) # dedup

        return new_prompts

    
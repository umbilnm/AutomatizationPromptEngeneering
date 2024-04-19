from langchain.prompts import PromptTemplate
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from utils.loader import strings, llm 
from typing import List

class Protegi:
    def evaluate_batch(self, df:pd.DataFrame, prompt:PromptTemplate, pred_col:str, true_col:str, text_col:str) -> pd.DataFrame:
        template = strings['templates']['prediction_template']
        template = PromptTemplate(template=template, input_variables=['message'])
        messages = df['messages']
        preds = []
        for message in tqdm(messages):
            prediction = llm.invoke(template.format(message=message)).content
            preds.append(prediction)
        df['predictions'] = 'ham' if prediction=='No' else 'spam'

        return df[text_col].values, df[true_col].values, df[pred_col].values, 
    
    def _sample_errors_string(self, true_val:np.array, pred_val:np.array, text_val:np.array)-> str:
        indexes = np.where(true_val != pred_val)[0]
        mapping = {0:'ham', 1:'spam'}
        mistakes_template = '''There are examples where models answer is incorrect:\n'''
        for idx in indexes:
            mistakes_template += text_val[idx] + '\n'
            mistakes_template +=f'True: {mapping[true_val[idx]]}\nPredicted: {mapping[pred_val[idx]]}\n'
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
        return answer, mistakes_template
    
    def generate_synonyms(self, prompt_section):
        """ Generate synonyms for a prompt section."""
        rewriter_prompt = f"Generate a variation of the following instruction while keeping the semantic meaning.\n\nInput: {prompt_section}\n\nOutput:"
        new_instructions = llm.invoke(rewriter_prompt)
        new_instructions = [x for x in new_instructions if x]
        return new_instructions
    
    
    def apply_gradient(self, prompt, error_str:str, feedback_str:str, steps_per_gradient:int):
        """ Incorporate feedback gradient into a prompt."""
        transformation_prompt = strings['templates']['transformation_template']  
        res = llm.invoke(transformation_prompt.format(prompt=prompt, \
                                               error_str=error_str, feedback_str=feedback_str, steps_per_gradient=steps_per_gradient))
        new_prompts = []
        for r in res:   
            new_prompts += self.parse_tagged_text(r.content, "<START>", "<END>")
        return new_prompts


    def expand_candidates(self, prompts, df, batch_size=10):
        """ Expand a list of prompts by generating gradient-based successors and 
            synonyms for each section.
        """
        minibatch = df.sample(batch_size)

        new_prompts = []
        for prompt in tqdm(prompts, desc=f'expanding {len(prompts)} prompts'):

            # evaluate prompt on minibatch
            texts, labels, preds = self.evaluate_batch(minibatch, prompt)
            error_string = self._
            # get gradients
            new_task_sections = []
        
            error_str = self._sample_errors_string(true_val=labels, pred_val=preds, text_val=texts)
            gradients = self._get_gradient(error_str)
            new_task_sections = []
            for feedback, error_string in tqdm(gradients, desc='applying gradients'):
                tmp = self.apply_gradient(error_string, feedback, self.opt['steps_per_gradient'])
                new_task_sections += tmp

            # generate synonyms
            mc_sampled_task_sections = []
            if self.opt['mc_samples_per_step'] > 0:
                for sect in tqdm(new_task_sections, desc='mc samples'):
                    mc_sects = self.generate_synonyms(sect)
                    mc_sampled_task_sections += mc_sects

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


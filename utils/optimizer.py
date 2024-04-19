class Protegi:
    def __init__(self):
        pass

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

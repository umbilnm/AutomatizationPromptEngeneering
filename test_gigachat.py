import pandas as pd
import os
from utils.loader import strings
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.gigachat import GigaChat
from utils.optimizer import Protegi
load_dotenv()
TOKEN = os.getenv("GIGACHAT_TOKEN")
GIGACHAT_AUTH = os.environ.get('GIGACHAT_AUTH')
chat = GigaChat(credentials=f'{GIGACHAT_AUTH}', verify_ssl_certs=False)
prompt = strings['templates']['spam']['prediction_template']
task = 'spam'
eval_dataset = pd.read_csv(f'./data/val_{task}.csv')
train_dataset = pd.read_csv(f'./data/train_{task}.csv')
sample = train_dataset.sample(10)
prot = Protegi('spam', 'gigachat', chat)
texts, labels, preds = prot.evaluate_batch()
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from  ruamel.yaml import YAML
load_dotenv()

PROXY_API_KEY = os.getenv("PROXY_API_KEY")


emb_fn = OpenAIEmbeddings(
    api_key=os.getenv("PROXY_API_KEY"),
    model="text-embedding-ada-002",
    base_url="https://api.proxyapi.ru/openai/v1",
)

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-1106",
    temperature=0.7,
    api_key=PROXY_API_KEY,
    base_url="https://api.proxyapi.ru/openai/v1"
)


yaml = YAML()
with open('/home/umbilnm/python_ml/AutomatizationPromptEngeneering/artifacts/strings.yaml', "r", encoding="utf-8") as file:
    strings = yaml.load(file) or {}
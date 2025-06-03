
from langchain_ollama import ChatOllama as Ollama
# from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
import json, os

store = {}

def track_queries_on_langsmith(flag: bool = False):
    if flag:
        pass_keys = get_config("key.json")
        os.environ["LANGCHAIN_API_KEY"] = pass_keys["LANGCHAIN_API_KEY"]
        os.environ["LANGCHAIN_TRACING_V2"] = pass_keys["LANGCHAIN_TRACING_V2"]
        os.environ["LANGCHAIN_PROJECT"] = pass_keys["LANGCHAIN_PROJECT_NAME"]

def fetch_llm(inference_type: str, groq_api_key: str = "") -> tuple:
    if inference_type == "paid":
        # Initialize the Groq Chat model : Paid inference
        llm = ChatGroq(model="Gemma2-9b-It", api_key=groq_api_key)
        model_name = "Gemma2-9b-It - GROQ"
    else:
        # Ollama model initialization
        llm = Ollama(model="llama3.2")
        model_name = "Llama 3.2 - Ollama"
    
    return llm, model_name

def get_config(path: str) -> dict:
    """
    Load configuration from a JSON file.
    """
    with open(path, 'r') as file:
        config = json.load(file)
    return config

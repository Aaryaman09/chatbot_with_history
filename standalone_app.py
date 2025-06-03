import json, os
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from utils import fetch_llm, get_config, track_queries_on_langsmith

class ChatBotOllama:
    """
    A simple chatbot that uses Ollama or Groq.
    """
    def __init__(self, pass_keys: dict=get_config("key.json")):
        self.pass_keys = pass_keys
        self.store = {}
        # Initialize tracking on LangSmith 
        track_queries_on_langsmith(flag=pass_keys.get("track_queries_on_langsmith", False))

    def get_session_history(self, session_id:str) -> BaseChatMessageHistory:
        """
        Get the session chat message history.
        """
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def fetch_message_history_runnable(self):
        # fetching the LLM from utils
        llm, model_name = fetch_llm(self.pass_keys["llm_service"], self.pass_keys.get("GROQ_API_KEY"))

        with_message_history = RunnableWithMessageHistory(llm, self.get_session_history)

        return with_message_history

if __name__ == "__main__":
    chatbot = ChatBotOllama()
    with_message_history = chatbot.fetch_message_history_runnable()

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        key_config = get_config("key.json")
        configuration = {
            "configurable": {
                "session_id": key_config.get("chat_session_id"),
            }
        }

        # Invoke the LLM with the human message
        response = with_message_history.invoke([HumanMessage(content=user_input)], config=configuration)
        print("Assistant:", response.content)
        
        del key_config
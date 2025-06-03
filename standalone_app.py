import json, os
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import fetch_llm, get_config, get_session_history, track_queries_on_langsmith

pass_keys = get_config("key.json")
track_queries_on_langsmith(flag=pass_keys.get("track_queries_on_langsmith", False))

# fetching the LLM from utils
llm, model_name = fetch_llm(pass_keys["llm_service"], pass_keys.get("GROQ_API_KEY"))

with_message_history = RunnableWithMessageHistory(llm, get_session_history)

if __name__ == "__main__":
    # Define the function to handle the translation
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
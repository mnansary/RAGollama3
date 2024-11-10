import streamlit as st
import time

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager



# Initialize session state

if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, designed to assist users with their inquiries in a detailed and informative manner. 
                                Ensure your tone is professional, yet approachable, and remember to communicate in Bengali (বাংলা).

                                   User:{question} 
                                   Chatbot:"""

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=[ "question"],
        template=st.session_state.template,
    )




# Initialize LLM

if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(
        base_url="http://localhost:11434",
        model="llama3.1:70b",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )




st.title("Chatbot - Conversational AI")




# Handle user input

if user_input := st.chat_input("You:", key="user_input"):
    user_message = {"role": "user", "message": f"{user_input}" }
    
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for chunk in st.session_state.llm.stream(f"{user_input}"):
            full_response += chunk
            message_placeholder.markdown(full_response)
        
        chatbot_message = {"role": "assistant", "message": full_response}
    
else:
    st.write("Please enter your query to start the chatbot")
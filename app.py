import streamlit as st
import os
import time

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from embedding import CustomEmbeddings
from config import *

# Add embedding function
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = CustomEmbeddings(EMBEDDING_MODEL)

# Ensure directories exist
if not os.path.exists(VECTOR_DB_PATH):
    st.error("Vector store directory not found. Please run the preprocessing script.")

# Initialize session state for prompt template (without history)
if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, designed to assist users with their inquiries in a detailed and informative manner. 
                                Your responses should not only answer the user's questions but also provide additional context, relevant examples, and insights related to the topic at hand. 
                                Ensure your tone is professional, yet approachable, and remember to communicate in Bengali (বাংলা).

                                Context: {context}

                                User: {question}
                                Chatbot: Please provide a thorough response, including any relevant details or explanations that might help the user better understand the topic.
                                """

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["context", "question"],  # Expect "query" instead of "question"
        template=st.session_state.template,
    )

# Initialize vectorstore and LLM
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=st.session_state.embedding_model
    )

if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(
        base_url=MODEL_BASE_URL,
        model=MODEL_NAME,
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("Chatbot - to talk to your Database")

# Initialize retriever
st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 1})

# Initialize QA chain without memory, only retrieving context
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=st.session_state.llm,
        chain_type='stuff',
        retriever=st.session_state.retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": st.session_state.prompt,
        }
    )
    
# Display the chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["message"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["message"])

# Handle user input
if user_input := st.chat_input("You:", key="user_input"):
    user_message = {"role": "user", "message": user_input}
    st.session_state.chat_history.append(user_message)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get the response from the QA chain (only based on retrieved context)
    with st.chat_message("assistant"):
        with st.spinner("Assistant is typing..."):
            response = st.session_state.qa_chain(user_input)
        message_placeholder = st.empty()
        full_response = response['result']
        message_placeholder.markdown(full_response)

    chatbot_message = {"role": "assistant", "message": full_response}
    st.session_state.chat_history.append(chatbot_message)
else:
    st.write("Please enter your query to start the chatbot.")

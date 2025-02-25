# streamlit_app.py (Modified to use API)
import streamlit as st
import requests
import json

# API endpoint URL (adjust if your Flask app is running elsewhere)
API_URL = "http://0.0.0.0:3030/chat"

st.title("Bangla RAG Chatbot (API Backend)")

# Initialize session state for chat history if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# def clear_chat_history():
#     st.session_state.chat_history = []
#     # No need to clear backend context from frontend anymore, backend manages it


# # Clear chat history button
# if st.button("Clear Conversation"):
#     clear_chat_history()
#     st.experimental_rerun()


# Chat input
user_question = st.text_input("আপনার প্রশ্ন এখানে লিখুন:", key="user_question")

if user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Call the Flask API endpoint
            with requests.post(API_URL, json={"question": user_question}, stream=True) as response:
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            chunk = line.decode("utf-8")
                            full_response += chunk
                            message_placeholder.write(full_response + "▌")  # Update live
                    message_placeholder.write(full_response)  # Final display
                else:
                    full_response = f"Error from backend API: {response.status_code} - {response.text}"
                    message_placeholder.write(full_response)

        except requests.exceptions.RequestException as e:
            full_response = f"Error connecting to backend API: {e}"
            message_placeholder.write(full_response)


    st.session_state.chat_history.append({"role": "assistant", "content": full_response})


# Display chat history
for chat_message in st.session_state.chat_history:
    with st.chat_message(chat_message["role"]):
        st.write(chat_message["content"])
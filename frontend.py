import streamlit as st
import requests

API_URL = "http://0.0.0.0:3030/query/"

st.title("Bangla Chatbot (RAG)")
st.write("Ask a question and get a streamed response.")

# User input field
question = st.text_input("আপনার প্রশ্ন লিখুন:")

if st.button("জিজ্ঞাসা করুন") and question:
    with st.empty():  # Create a placeholder for streaming
        response = requests.get(API_URL, params={"question": question}, stream=True)

        if response.status_code == 200:
            answer = ""
            for chunk in response.iter_content(chunk_size=50):
                if chunk:
                    text = chunk.decode("utf-8")
                    answer += text
                    st.write(answer)  # Update the text dynamically
        else:
            st.error("সার্ভার সমস্যা হয়েছে। পরে চেষ্টা করুন।")

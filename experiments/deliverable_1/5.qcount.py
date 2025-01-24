MODEL_NAME="llama3.1:70b"       
MODEL_BASE_URL='http://localhost:11434'
JSON_DIR="wikiquestions_cleaned"
SAVE_DIR="wiki_qna"

import os
import time
from tqdm import  tqdm 
import json
from glob import glob
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama

template = """You are a knowledgeable chatbot, designed to assist users with their inquiries in a detailed and informative manner. 
            Your responses should only answer the user's questions . 
            Ensure your tone is professional, yet approachable, and remember to communicate in Bengali (বাংলা).

            Context: {}

            User: {}
            Chatbot: Please provide precise humanly response.
            """

llm = Ollama(
        base_url=MODEL_BASE_URL,
        model=MODEL_NAME
    )


all_jsons=[os.path.basename(jsonf) for jsonf in tqdm(glob(os.path.join(JSON_DIR,"*.json")))]
generated_jsons=[os.path.basename(jsonf) for jsonf in tqdm(glob(os.path.join(SAVE_DIR,"*.json")))]
print(len(all_jsons))
print(len(generated_jsons))
all_jsons=[f for f in all_jsons if f not in generated_jsons]
print(len(all_jsons))
ck=0
for q_file in tqdm(generated_jsons):
    q_file=os.path.join(SAVE_DIR,q_file)
    with open(q_file, 'r', encoding='utf-8') as file:
        entry = json.load(file)
    ck+=entry["num_entries"]
print(ck)
    # context=entry["text"]
    # questions=entry["question"]
    # results=[]
    # for question in questions:
    #     prompt=template.format(context,question)
    #     results.append(llm(prompt))
    
    # entry["answer"]=results
    # output_file =  f"wiki_qna/{os.path.basename(q_file)}"
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(entry, f, ensure_ascii=False, indent=2)

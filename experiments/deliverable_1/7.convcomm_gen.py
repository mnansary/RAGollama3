MODEL_NAME="llama3.1:70b"       
MODEL_BASE_URL='http://localhost:11434'
QNA_JSON_DIR="wikiqna"
ID_JSON_DIR="wiki_intent_dialouge"
SAVE_DIR="BanConvComm"

import os
import time
from tqdm import  tqdm 
import json
from glob import glob
all_jsons=[os.path.basename(jsonf) for jsonf in tqdm(glob(os.path.join(QNA_JSON_DIR,"*.json")))]

import os
import time
from tqdm import  tqdm 
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama

template=""" 
Prompt:
You are an AI assistant that extracts the intent from a given question or command. 
The intent is represented as a hierarchical label in the format <category.subcategory[.sub-subcategory]>, where each level provides more specificity about the intent. 
Use the provided examples as a guide:

Examples:
Input: ব্যাংকের ব্র্যাঞ্চ অফিসের ঠিকানা দিন।
Output: bank.address

Input: পোস্ট অফিসের এরিয়া কী?
Output: post_office.area

Input: আপনার কোম্পানির ইমেল আইডি কী?
Output: company.email

Input: বিমানবন্দরের যাত্রী পরিষেবা নাম্বারটি কী?
Output: airport.passenger_service.phone

Input: মেডিকেল কলেজের প্রধান অধ্যাপকের নাম কী?
Output: medical_college.professor.name

Task:
Given a new input, extract its hierarchical intent in the format described. DO NOT PRINT ADDIONALT TEXT OR THE QUESTION ITSELF. JUST THE INTENT.

Input:{}
"""

llm = Ollama(
        base_url=MODEL_BASE_URL,
        model=MODEL_NAME
    )


for q_file in tqdm(all_jsons):
    i_file=os.path.join(ID_JSON_DIR,q_file)
    q_file=os.path.join(QNA_JSON_DIR,q_file)
    data={}
    with open(q_file, 'r', encoding='utf-8') as file:
        entry = json.load(file)
    
    questions=entry["question"]
    answers=entry["answer"]
    
    with open(i_file, 'r', encoding='utf-8') as file:
        ientry = json.load(file)

    if len(ientry["A"])==len(ientry["B"]) and len(ientry["A"])!=0:
        data["user"]=ientry["A"]
        data["response"]=ientry["B"]
    else:
        data["user"]=questions
        data["response"]=answers
    results=[]
    for question in questions:
        prompt=template.format(question)
        results.append(llm(prompt))
        
    data["intents"]=results
    data["num_entries"]=len(data["intents"])
    output_file=q_file.replace(QNA_JSON_DIR,SAVE_DIR)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
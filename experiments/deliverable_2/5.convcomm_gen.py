
import os
import time
from tqdm import  tqdm 
import json
from glob import glob
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

def process_intent(all_jsons):
    for i_file in tqdm(all_jsons):
        base=os.path.basename(i_file) 
        output_file=os.path.join(SAVE_DIR,base)
        if os.path.exists(output_file):
            continue
        
        data={}
        with open(i_file, 'r', encoding='utf-8') as file:
            ientry = json.load(file)
        data["user"]=ientry["A"]
        data["response"]=ientry["B"]
    
        entries=ientry["A"]
        results=[]
        for entry in entries:
            prompt=template.format(entry)
            results.append(llm(prompt))
            
        data["intents"]=results
        data["num_entries"]=len(data["intents"])


        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


if __name__=="__main__":
    MODEL_NAME="llama3.3"       
    MODEL_BASE_URL='http://localhost:11434'
    ID_JSON_DIR="/home/vpa/deliverables/D4/bccData"
    SAVE_DIR="/home/vpa/deliverables/BanConvComm"
    os.makedirs(SAVE_DIR,exist_ok=True)
    all_jsons=[jsonf for jsonf in tqdm(glob(os.path.join(ID_JSON_DIR,"*.json")))]

    llm = Ollama(
                base_url=MODEL_BASE_URL,
                model=MODEL_NAME,
                temperature=0.4, 
                num_ctx=4096,
                )
    process_intent(all_jsons)
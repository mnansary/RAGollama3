#------------------------------------------
# imports
#------------------------------------------
import os
import time
import json
import pandas as pd 
from tqdm import  tqdm 
tqdm.pandas()
import re

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
#------------------------------------------
# dirs
#------------------------------------------

SAVE_DIR="questions"
DATA_CSV="merged_processed-passages-20250101.csv"
MODEL_NAME="llama3.1:70b"       
MODEL_BASE_URL='http://localhost:11434'

llm = Ollama(
        base_url=MODEL_BASE_URL,
        model=MODEL_NAME
    )
#------------------------------------------
# functions
#------------------------------------------
def create_contexts(text, max_context_words=500):
    
    paragraphs = text.split('\n\n')
    # Generate contexts where the word count is >= max_context_words
    contexts = []
    current_context = ''
    current_word_count = 0

    for paragraph in paragraphs:
        word_count = len(str(paragraph).replace("\n","").strip().split())

        # If adding this paragraph keeps us below max_context_words, add it to the current context
        if current_word_count + word_count < max_context_words:
            current_context += (paragraph + '\n\n')
            current_word_count += word_count
        else:
            # If the current context is not empty, finalize it and start a new one
            if current_context:
                contexts.append(current_context.strip())
                current_context = ''
                current_word_count = 0

            # If the paragraph itself is >= max_context_words, add it as its own context
            if word_count >= max_context_words:
                contexts.append(paragraph.strip())
            else:
                current_context = paragraph + '\n\n'
                current_word_count = word_count

    # Add the last context if it exists
    if current_context.strip():
        contexts.append(current_context.strip())

    contexts=[re.sub(r'\n+', '\n', text) for text in contexts]
    contexts=[text.replace("\n","।") for text in contexts]
    contexts=[re.sub(r'\s+', ' ', text) for text in contexts]
    return contexts

template=""" 
You are an intelligent assistant. Your task is to analyze the given context and generate an appropriate number of questions and their answers in Bengali (বাংলা) based on the content's complexity, depth, and details. Follow these steps:

1. Carefully read and understand the context provided below.
2. Decide how many questions should be created based on the richness and intricacy of the context. Minimum 2 and Maximum 15
3. Generate the decided number of questions in Bengali. For each question, provide its answer as well. Ensure that:
   - The questions are relevant to the content.
   - The questions are varied in nature (e.g., factual, analytical, or thought-provoking).
   - Both the questions and answers are clear and concise.
4. Present your output in the following format:
   - "প্রশ্নের সংখ্যা: [X]"
   - For each question and answer pair:
     - "প্রশ্ন [Number]: [Question in Bengali]"
     - "উত্তর [Number]: [Answer in Bengali]"

**Example 1:**

**Context:**
প্রতিবছর ২১ ফেব্রুয়ারি আন্তর্জাতিক মাতৃভাষা দিবস পালন করা হয়। ১৯৫২ সালের ২১ ফেব্রুয়ারি বাংলা ভাষার অধিকারের জন্য আন্দোলনরত ছাত্ররা তাদের জীবন উৎসর্গ করেন। তাদের ত্যাগের স্মৃতিতে এই দিনটি পালন করা হয়।

**Result:**
প্রশ্নের সংখ্যা: ৩  
প্রশ্ন ১: আন্তর্জাতিক মাতৃভাষা দিবস কখন পালিত হয়?  
উত্তর ১: আন্তর্জাতিক মাতৃভাষা দিবস প্রতি বছর ২১ ফেব্রুয়ারি পালিত হয়।  

প্রশ্ন ২: ১৯৫২ সালের ২১ ফেব্রুয়ারিতে কী ঘটেছিল?  
উত্তর ২: ১৯৫২ সালের ২১ ফেব্রুয়ারিতে বাংলা ভাষার অধিকারের জন্য ছাত্ররা তাদের জীবন উৎসর্গ করেন।  

প্রশ্ন ৩: মাতৃভাষা দিবস কেন গুরুত্বপূর্ণ?  
উত্তর ৩: মাতৃভাষা দিবস গুরুত্বপূর্ণ কারণ এটি ভাষার অধিকারের জন্য আত্মত্যাগের স্মৃতি বহন করে।  


**Example 2:**

**Context:**
বাংলাদেশের প্রধান নদীগুলোর মধ্যে পদ্মা, মেঘনা এবং যমুনা উল্লেখযোগ্য। পদ্মা গঙ্গা নদীর একটি শাখা এবং যমুনা ব্রহ্মপুত্র নদীর প্রধান শাখা। এ নদীগুলোর উপর ভিত্তি করে দেশের কৃষি, পরিবহন এবং জীবিকা নির্ভরশীল।

**Result:**
প্রশ্নের সংখ্যা: ৪  
প্রশ্ন ১: বাংলাদেশের প্রধান নদীগুলোর নাম কী কী?  
উত্তর ১: বাংলাদেশের প্রধান নদীগুলো হলো পদ্মা, মেঘনা এবং যমুনা।  

প্রশ্ন ২: পদ্মা কোন নদীর শাখা?  
উত্তর ২: পদ্মা গঙ্গা নদীর একটি শাখা।  

প্রশ্ন ৩: যমুনা নদীর উৎস কী?  
উত্তর ৩: যমুনা ব্রহ্মপুত্র নদীর প্রধান শাখা।  

প্রশ্ন ৪: বাংলাদেশের অর্থনীতিতে নদীগুলোর ভূমিকা কী?  
উত্তর ৪: নদীগুলো দেশের কৃষি, পরিবহন এবং জীবিকার ক্ষেত্রে গুরুত্বপূর্ণ ভূমিকা পালন করে।  


### Task  
Decide the number of questions to generate based on the context and create both questions and their answers in Bengali.  
প্রসঙ্গটি থেকে প্রশ্নের সংখ্যা নির্ধারণ করুন এবং সেই সংখ্যক প্রশ্ন এবং উত্তর তৈরি করুন। সমস্ত প্রশ্ন এবং উত্তর বাংলায় লিখুন।  

**Context:**  
{}

"""



template2=""" 
You are an intelligent assistant. Your task is to analyze the given context and generate an appropriate number of questions and their answers in Bengali (বাংলা) based on the content's complexity, depth, and details. Follow these steps:

1. Carefully read and understand the context provided below.
2. Decide how many questions should be created based on the richness and intricacy of the context. Minimum 2 and Maximum 15
3. Generate the decided number of questions in Bengali. For each question, provide its answer as well. Ensure that:
   - The questions are relevant to the content.
   - The questions are varied in nature (e.g., factual, analytical, or thought-provoking).
   - Both the questions and answers are clear and concise.
4. Present your output in the following format:
   - "প্রশ্নের সংখ্যা: [X]"
   - For each question and answer pair:
     - "প্রশ্ন [Number]: [Question in Bengali]"
     - "উত্তর [Number]: [Answer in Bengali]"

**Example:**

**Context:**
প্রতিবছর ২১ ফেব্রুয়ারি আন্তর্জাতিক মাতৃভাষা দিবস পালন করা হয়। ১৯৫২ সালের ২১ ফেব্রুয়ারি বাংলা ভাষার অধিকারের জন্য আন্দোলনরত ছাত্ররা তাদের জীবন উৎসর্গ করেন। তাদের ত্যাগের স্মৃতিতে এই দিনটি পালন করা হয়।

**Result:**
প্রশ্নের সংখ্যা: ৩  
প্রশ্ন ১: আন্তর্জাতিক মাতৃভাষা দিবস কখন পালিত হয়?  
উত্তর ১: আন্তর্জাতিক মাতৃভাষা দিবস প্রতি বছর ২১ ফেব্রুয়ারি পালিত হয়।  

প্রশ্ন ২: ১৯৫২ সালের ২১ ফেব্রুয়ারিতে কী ঘটেছিল?  
উত্তর ২: ১৯৫২ সালের ২১ ফেব্রুয়ারিতে বাংলা ভাষার অধিকারের জন্য ছাত্ররা তাদের জীবন উৎসর্গ করেন।  

প্রশ্ন ৩: মাতৃভাষা দিবস কেন গুরুত্বপূর্ণ?  
উত্তর ৩: মাতৃভাষা দিবস গুরুত্বপূর্ণ কারণ এটি ভাষার অধিকারের জন্য আত্মত্যাগের স্মৃতি বহন করে।  

### Task  
Decide the number of questions to generate based on the context and create both questions and their answers in Bengali.  
প্রসঙ্গটি থেকে প্রশ্নের সংখ্যা নির্ধারণ করুন এবং সেই সংখ্যক প্রশ্ন এবং উত্তর তৈরি করুন। সমস্ত প্রশ্ন এবং উত্তর বাংলায় লিখুন।  

**Context:**  
{}

"""

template3=""" 
You are an intelligent assistant. Your task is to analyze the given context and generate an appropriate number of questions and their answers in Bengali (বাংলা) based on the content's complexity, depth, and details. Follow these steps:

1. Carefully read and understand the context provided below.
2. Decide how many questions should be created based on the richness and intricacy of the context. Minimum 2 and Maximum 15
3. Generate the decided number of questions in Bengali. For each question, provide its answer as well. Ensure that:
   - The questions are relevant to the content.
   - The questions are varied in nature (e.g., factual, analytical, or thought-provoking).
   - Both the questions and answers are clear and concise.
4. Present your output in the following format:
   - "প্রশ্নের সংখ্যা: [X]"
   - For each question and answer pair:
     - "প্রশ্ন [Number]: [Question in Bengali]"
     - "উত্তর [Number]: [Answer in Bengali]"

### Task  
Decide the number of questions to generate based on the context and create both questions and their answers in Bengali.  
প্রসঙ্গটি থেকে প্রশ্নের সংখ্যা নির্ধারণ করুন এবং সেই সংখ্যক প্রশ্ন এবং উত্তর তৈরি করুন। সমস্ত প্রশ্ন এবং উত্তর বাংলায় লিখুন।  

**Context:**  
{}

"""



additional=""" 

"""


df=pd.read_csv(DATA_CSV)
df = df[df['text'].notna()]
df=df[["text",'site_name','passage_heading']]
df["passage_id"]=df.index+1
df["text"]=df.progress_apply(lambda x: x["text"].replace("site_name:"+x["site_name"],"").replace("passage_heading:"+x["passage_heading"],"")[2:],axis=1)
df.dropna(inplace=True)
# texts=[]
# pids=[]
# sites=[]
# for site in df.site_name.unique():
#     sdf=df.loc[df.site_name==site]
#     text=sdf["text"].tolist()
#     text="#|passage|#".join(text)
#     texts.append(text)
#     sites.append(site)
#     pids.append(sdf.passage_id.tolist())

# df=pd.DataFrame({"text":texts,"topic":sites,"passage_ids":pids})

def try1(data):
    prompt=template.format(data)
    result=llm(prompt)
    # Extract questions
    questions = re.findall(r"প্রশ্ন \d+: (.*?)(?:\n|$)", result)

    # Extract answers
    answers = re.findall(r"উত্তর \d+: (.*?)(?:\n|$)", result)
    if len(questions)>=2:
        return result,questions,answers
    else:
        return None,None,None

def try2(data):
    prompt=template2.format(data)
    result=llm(prompt)
    # Extract questions
    questions = re.findall(r"প্রশ্ন \d+: (.*?)(?:\n|$)", result)

    # Extract answers
    answers = re.findall(r"উত্তর \d+: (.*?)(?:\n|$)", result)
    if len(questions)>=2:
        return result,questions,answers
    else:
        return None,None,None
    
def try3(data):
    prompt=template3.format(data)
    result=llm(prompt)
    # Extract questions
    questions = re.findall(r"প্রশ্ন \d+: (.*?)(?:\n|$)", result)

    # Extract answers
    answers = re.findall(r"উত্তর \d+: (.*?)(?:\n|$)", result)
    if len(questions)>=2:
        return result,questions,answers
    else:
        return None,None,None

        

def generate(data):
    result,questions,answers=try1(data)
    if result is None:
        result,questions,answers=try2(data)
        if result is None:
            result,questions,answers=try3(data)
            return result,questions,answers
        else:
            return result,questions,answers
    else:
        return  result,questions,answers   



for idx in tqdm(range(len(df))):
    text=df.iloc[idx,0]
    info=df.iloc[idx,1]
    contexts=create_contexts(text,max_context_words=150)
    for cidx,context in enumerate(contexts):
        entry={}
        entry["text"]=str(info)+"\n\n"+str(context)
        result,questions,answers=generate(entry["text"])
        entry["response"]=result
        entry["question"]=questions
        entry["answer"]=answers
        
        if result is not None:
            output_file =  f"{SAVE_DIR}/{idx}_{cidx}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
        else:
            output_file =  f"{SAVE_DIR}/{idx}_{cidx}_wrong.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
            
            


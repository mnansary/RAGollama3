


template=""" 
### Instructions
You are an intelligent assistant. Your task is to analyze the given context and generate an appropriate number of questions and their answers in Bengali (বাংলা) based on the content's complexity, depth, and details. Follow these steps:
Decide the number of questions to generate based on the context and create both questions and their answers in Bengali.  
প্রসঙ্গটি থেকে প্রশ্নের সংখ্যা নির্ধারণ করুন এবং সেই সংখ্যক প্রশ্ন এবং উত্তর তৈরি করুন। সমস্ত প্রশ্ন এবং উত্তর বাংলায় লিখুন। 

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


### Example 1

### Context:
প্রতিবছর ২১ ফেব্রুয়ারি আন্তর্জাতিক মাতৃভাষা দিবস পালন করা হয়। ১৯৫২ সালের ২১ ফেব্রুয়ারি বাংলা ভাষার অধিকারের জন্য আন্দোলনরত ছাত্ররা তাদের জীবন উৎসর্গ করেন। তাদের ত্যাগের স্মৃতিতে এই দিনটি পালন করা হয়।

### Response
প্রশ্নের সংখ্যা: ৩  
প্রশ্ন ১: আন্তর্জাতিক মাতৃভাষা দিবস কখন পালিত হয়?  
উত্তর ১: আন্তর্জাতিক মাতৃভাষা দিবস প্রতি বছর ২১ ফেব্রুয়ারি পালিত হয়।  

প্রশ্ন ২: ১৯৫২ সালের ২১ ফেব্রুয়ারিতে কী ঘটেছিল?  
উত্তর ২: ১৯৫২ সালের ২১ ফেব্রুয়ারিতে বাংলা ভাষার অধিকারের জন্য ছাত্ররা তাদের জীবন উৎসর্গ করেন।  

প্রশ্ন ৩: মাতৃভাষা দিবস কেন গুরুত্বপূর্ণ?  
উত্তর ৩: মাতৃভাষা দিবস গুরুত্বপূর্ণ কারণ এটি ভাষার অধিকারের জন্য আত্মত্যাগের স্মৃতি বহন করে।  


### Example 2

### Context:
বাংলাদেশের প্রধান নদীগুলোর মধ্যে পদ্মা, মেঘনা এবং যমুনা উল্লেখযোগ্য। পদ্মা গঙ্গা নদীর একটি শাখা এবং যমুনা ব্রহ্মপুত্র নদীর প্রধান শাখা। এ নদীগুলোর উপর ভিত্তি করে দেশের কৃষি, পরিবহন এবং জীবিকা নির্ভরশীল।

### Response
প্রশ্নের সংখ্যা: ৪  
প্রশ্ন ১: বাংলাদেশের প্রধান নদীগুলোর নাম কী কী?  
উত্তর ১: বাংলাদেশের প্রধান নদীগুলো হলো পদ্মা, মেঘনা এবং যমুনা।  

প্রশ্ন ২: পদ্মা কোন নদীর শাখা?  
উত্তর ২: পদ্মা গঙ্গা নদীর একটি শাখা।  

প্রশ্ন ৩: যমুনা নদীর উৎস কী?  
উত্তর ৩: যমুনা ব্রহ্মপুত্র নদীর প্রধান শাখা।  

প্রশ্ন ৪: বাংলাদেশের অর্থনীতিতে নদীগুলোর ভূমিকা কী?  
উত্তর ৪: নদীগুলো দেশের কৃষি, পরিবহন এবং জীবিকার ক্ষেত্রে গুরুত্বপূর্ণ ভূমিকা পালন করে।  


### Context:
{}
### Response:
"""



import os
import sqlite3
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "BanglaLLM/BanglaLLama-3-8b-BnWiki-Instruct" 

#----------------------models--------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    load_in_4bit=False,
    device_map="auto",
    use_cache=True,
)

def generate_response(input_text):
    prompt = template.format(input_text)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    #print(len(inputs['input_ids']))
    generated_ids = model.generate(
        **inputs,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=(inputs['input_ids'].shape[-1])//10,
        do_sample=False,
        num_beams=1,
    )

    # Decode the generated output
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    response = outputs[0].split("### Response:")[-1].strip()
    return response
#------------------------------------llm functions-------------------------------


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

#------------------------------------------
# dirs
#------------------------------------------

SAVE_DIR="questions"
DATA_CSV="merged_processed-passages-20250101.csv"

#------------------------------------------
# functions
#------------------------------------------
def create_contexts(text, max_context_words=500):
    # Split the text into paragraphs first
    paragraphs = text.split('\n\n')
    contexts = []
    current_context = ''
    current_word_count = 0

    for paragraph in paragraphs:
        # Split paragraph into sentences by "।"
        sentences = re.split(r'।', paragraph)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:  # Skip empty sentences
                continue
            
            # Count words in the sentence
            word_count = len(sentence.split())

            # If adding this sentence keeps us below max_context_words, add it
            if current_word_count + word_count < max_context_words:
                current_context += (sentence + "। ")
                current_word_count += word_count
            else:
                # Finalize the current context
                if current_context.strip():
                    contexts.append(current_context.strip())
                
                # Start a new context with the current sentence
                current_context = sentence + "। "
                current_word_count = word_count

    # Add the last context if it exists
    if current_context.strip():
        contexts.append(current_context.strip())

    # Cleanup the contexts: normalize spaces and remove excessive newlines
    contexts = [re.sub(r'\s+', ' ', context).strip() for context in contexts]
    return contexts



df=pd.read_csv(DATA_CSV)
df = df[df['text'].notna()]
df=df[["text",'site_name','passage_heading']]
df["passage_id"]=df.index+1
df["text"]=df.progress_apply(lambda x: x["text"].replace("site_name:"+x["site_name"],"").replace("passage_heading:"+x["passage_heading"],"")[2:],axis=1)
df.dropna(inplace=True)

for idx in tqdm(range(len(df))):
    torch.cuda.empty_cache()
    text=df.iloc[idx,0]
    info=df.iloc[idx,1]
    contexts=create_contexts(text,max_context_words=150)
    for cidx,context in enumerate(contexts):
        output_file =  f"{SAVE_DIR}/{idx}_{cidx}.json"
        if os.path.exists(output_file):
            #print(output_file)
            continue
        if not os.path.exists(output_file):
            entry={}
            entry["text"]=str(info)+"\n\n"+str(context)
            #print(entry)
            result=generate_response(entry["text"])
            entry["response"]=result
            # Extract questions
            questions = re.findall(r"প্রশ্ন \d+: (.*?)(?:\n|$)", result)

            # Extract answers
            answers = re.findall(r"উত্তর \d+: (.*?)(?:\n|$)", result)
            entry["question"]=questions
            entry["answer"]=answers
            if len(questions)>=1:
                output_file =  f"{SAVE_DIR}/{idx}_{cidx}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(entry, f, ensure_ascii=False, indent=2)
            else:
                output_file =  f"{SAVE_DIR}/{idx}_{cidx}_wrong.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(entry, f, ensure_ascii=False, indent=2)
            
        
        
import os
import time
from tqdm import  tqdm 
import json
from glob import glob
import os
import time
from tqdm import  tqdm 
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
import pandas as pd 
import argparse
from openai import OpenAI

template = """You are tasked with converting a set of question-answer pairs into a natural conversation format. 
            The conversation should feel fluid, logical, and natural, as if two people are discussing the topics informally. 
            Here an example input data:

            A:বাংলা ভাষার সাথে সম্পর্কিত কোন দেশগুলির সংখ্যা কত?\n
            B:বাংলা ভাষার সাথে সম্পর্কিত ১০টি দেশগুলির নাম হচ্ছে:\n\n           1. বাংলাদেশ\n           2. ভারত\n           3. ইউনাইটেড কিংডম\n           4. যুক্তরাষ্ট্র \n           5. পাকিস্তান\n           6. ওমান\n           7. ইন্দোনেশিয়া\n           8. ব্রুনেই\n           9. মালয়েশিয়া\n           10. সিংগাপুর\n
            A:বাংলা ভাষার সাথে কোন ধরনের ভাষা সম্পর্কিত?\n
            B:Indo-Aryan languages\n
            A:বাংলা ভাষার অবস্থান কোন দেশে?\n      
            B:পৃথিবীর জনসংখ্যা অনুযায়ী ভাষা হিসেবে মাতৃভাষা হিসেবে বক্তৃতা ও লিখিত রূপে, চতুর্থ এবং শীর্ষস্থানে অধিষ্ঠিত হয়ে বাংলা ভাষার অনন্য অবস্থান।\n 
        
            Your goal is to convert these into a natural conversation in Bengali(বাংলা). For example, it might look something like this:

            A: বাংলা ভাষার সাথে সম্পর্কিত কোন দেশগুলির কথা শুনেছো?
            B: বাংলা ভাষার সাথে সম্পর্কিত ১০টি দেশ আছে। যেমন: বাংলাদেশ, ভারত, ইউনাইটেড কিংডম, যুক্তরাষ্ট্র, পাকিস্তান, ওমান, ইন্দোনেশিয়া, ব্রুনেই, মালয়েশিয়া, এবং সিংগাপুর।
            A: ওহ, বাংলা ভাষার সাথে কি ধরনের ভাষা সম্পর্কিত?
            B: বাংলা ভাষা ইন্দো-আর্য ভাষাগুলোর একটি অংশ।
            A: সত্যি? তাহলে বাংলা ভাষার অবস্থান সম্পর্কে কিছু বলো।
            B: অবশ্যই! জনসংখ্যার ভিত্তিতে, বাংলা ভাষা পৃথিবীর অন্যতম বৃহৎ ভাষা। এটি মাতৃভাষা হিসেবে চতুর্থ স্থান দখল করে আছে।
            Generate the output in a similar conversational format, ensuring the tone is friendly and informal while still conveying the original information accurately.

            Now convert the following input
            Input: {}
            
            """


def generate(prompt,client) -> str:
    completion = client.chat.completions.create(extra_body={},model="meta-llama/llama-3.3-70b-instruct:free",
                                                messages=[{"role": "user","content": prompt}],
                                                temperature=0.2)
    result = completion.choices[0].message.content
    return result

def generate_conv(df,client):
    all_jsons=df.paths.tolist()
    for q_file in tqdm(all_jsons):
        # check if it exists
        base=os.path.basename(q_file)
        sfile=os.path.join(SAVE_DIR,base)
        if os.path.exists(sfile):
            continue

        with open(q_file, 'r', encoding='utf-8') as file:
            entry = json.load(file)
        
        questions=entry["question"]
        answers=entry["answer"]

        input_data=""
        for q,a in zip(questions,answers):
            q_cleaned = q.replace("\n", "")  # Clean newline characters
            a_cleaned = a.replace("\n", "")  # Clean newline characters
            input_data += f'A:{q_cleaned}\nB:{a_cleaned}\n'
        prompt=template.format(input_data)
        try:
            conversation=generate(prompt,client)
        except Exception as e:
            continue
        # Initialize a dictionary to store conversation by speakers
        conversation_dict = {"A": [], "B": []}

        # Split the conversation into lines and process each line
        for line in conversation.strip().split("\n"):
            # Split the line into speaker and statement
            if ": " in line:
                speaker, statement = line.split(": ", 1)
                if speaker in conversation_dict:
                    conversation_dict[speaker].append(statement.strip())
        conversation_dict["input"]=input_data
        conversation_dict["response"]=conversation
        conversation_json = json.dumps(conversation_dict, ensure_ascii=False, indent=2)
        
        with open(sfile, "w", encoding="utf-8") as file:
            file.write(conversation_json)
        # print("Done")
        # break

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Generate conv comm from JSON files with specified index"
    )
    
    # Add idx argument
    parser.add_argument(
        '--idx',
        type=int,
        default=0,
        help='Index value for processing (default: 0)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Use the parsed idx value
    idx = args.idx
    
    # You can now use idx in your code
    print(f"Running with idx = {idx}")

    df=pd.read_csv(f"/home/vpa/RAGollama3/data/chunks/{idx}.csv")
    apis=pd.read_csv("/home/vpa/RAGollama3/data/chunks/ansary_apis.csv")
    api_key=apis["api"].tolist()[idx]
    # create client
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    )
    
    SAVE_DIR="/home/vpa/deliverables/bccData"
    os.makedirs(SAVE_DIR,exist_ok=True)

    
    df=pd.read_csv(f"/home/vpa/RAGollama3/data/chunks/{idx}.csv") 
    # Generate Q&A for the filtered DataFrame
    generate_conv(df,client)
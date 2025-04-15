import pandas as pd
import json
import os
import re
import time
import sys
from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm
import argparse
import threading
import queue
from langchain_community.llms import Ollama


llm = Ollama(model="llama3.3", num_ctx=32768)

#-----------------------------------------------------------------------------------
template = """ 
You are an intelligent assistant. 
Your task is to analyze the given context and generate an appropriate number of questions and their answers in Bengali (বাংলা) based on the content's complexity, depth, and details. 
Follow these steps:

1. Carefully read and understand the context provided below.
2. Decide how many questions should be created based on the richness and intricacy of the context. Minimum 5 and Maximum 15
3. Generate the decided number of questions in Bengali. For each question, provide its answer as well. Ensure that:
   - The questions are relevant to the content.
   - The questions are varied in nature (e.g., factual, analytical, or thought-provoking).
   - Both the questions and answers are clear and concise.
4. Present your output in the following format:
   - For each question and answer pair:
     - "প্রশ্ন [Number]: [Question in Bengali]"
     - "উত্তর [Number]: [Answer in Bengali]"

**Example 1:**

**Context:**
প্রতিবছর ২১ ফেব্রুয়ারি আন্তর্জাতিক মাতৃভাষা দিবস পালন করা হয়। ১৯৫২ সালের ২১ ফেব্রুয়ারি বাংলা ভাষার অধিকারের জন্য আন্দোলনরত ছাত্ররা তাদের জীবন উৎসর্গ করেন। তাদের ত্যাগের স্মৃতিতে এই দিনটি পালন করা হয়।

**Result:**
প্রশ্ন ১: আন্তর্জাতিক মাতৃভাষা দিবস কখন পালিত হয়?  
উত্তর ১: আন্তর্জাতিক মাতৃভাষা দিবস প্রতি বছর ২১ ফেব্রুয়ারি পালিত হয়।  

প্রশ্ন ২: ১৯৫২ সালের ২১ ফেব্রুয়ারিতে কী ঘটেছিল?  
উত্তর ২: ১৯৫২ সালের ২১ ফেব্রুয়ারিতে বাংলা ভাষার অধিকারের জন্য ছাত্ররা তাদের জীবন উৎসর্গ করেন।  

প্রশ্ন ৩: মাতৃভাষা দিবস কেন গুরুত্বপূর্ণ?  
উত্তর ৩: মাতৃভাষা দিবস গুরুত্বপূর্ণ কারণ এটি ভাষার অধিকারের জন্য আত্মত্যাগের স্মৃতি বহন করে।  

### Task  
Decide the number of questions to generate based on the context and create both questions and their answers in Bengali.  

**Context:**  
{}
"""

#------------------------------------------------------------------------------------------------------------------

def generate_questions_answers(context: str,client) -> str:
    """
    Generate questions and answers in Bengali based on the given context with a progress indicator.

    Args:
        context (str): The context to analyze and generate Q&A from.

    Returns:
        str: A string containing the generated questions and answers in the specified format.
    """
    prompt = template.format(context)
    
    completion = client.chat.completions.create(extra_body={},model="meta-llama/llama-3.3-70b-instruct:free",
                                                messages=[{"role": "user","content": prompt}])
    # Generate Q&A using the LLM
    result = completion.choices[0].message.content
    
    
    return result

#------------------------------------------------------------------------------------------------------------------

# def get_total_questions(save_dir: str) -> int:
#     """Calculate total questions from all JSON files in save_dir."""
#     total_questions = 0
#     if os.path.exists(save_dir):
#         for filename in os.listdir(save_dir):
#             if filename.endswith('.json'):
#                 try:
#                     with open(os.path.join(save_dir, filename), 'r', encoding='utf-8') as f:
#                         data = json.load(f)
#                         total_questions += len(data["question"])
#                 except Exception:
#                     continue  # Skip problematic files
#     return total_questions

def gen_qna(df: pd.DataFrame, save_dir: str, client, MAX_GEN=120000) -> None:
    """
    Process a DataFrame of JSON file paths to generate Q&A entries and save them as JSON files.

    Args:
        df (pd.DataFrame): DataFrame containing a 'paths' column with JSON file paths.
        save_dir (str): Directory to save the generated Q&A JSON files.
        client: Client object for generating questions and answers
        MAX_GEN (int): Maximum number of questions allowed across all runs
    """
    # Initialize counters
    #overall_count = get_total_questions(save_dir)  # Initial total from existing files
    data_count = 0  # Counter for processed entries in current run
    questions_count = 0  # Questions generated in current run
    total_gen_time = 0  # Total time for generation and processing
    gen_count = 0  # Count of generation operations
    
    # Time tracking for 60-second updates
    last_update_time = time.time()
    UPDATE_INTERVAL = 60  # 60 seconds
    
    # Create tqdm progress bar for current run
    with tqdm(total=len(df['paths']), desc="Processing Q&A") as pbar:
        # Iterate over each JSON file path in the DataFrame
        for json_path in df['paths'].tolist():
            # if overall_count >= MAX_GEN:
            #     print("FINISHED PROCESSING REQUIRED DATA")
            #     break

            start_time = time.time()  # Record start time for this entry
            
            try:
                with open(json_path, "r", encoding='utf-8') as f:
                    passage = json.load(f)
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                data_count += 1
                pbar.update(1)
                continue
            
            _id = passage["meta"]["id"]
            save_json = os.path.join(save_dir, f"{_id}.json")
            
            # Check if the Q&A file already exists
            if not os.path.exists(save_json):
                entry: Dict[str, Any] = {}  # Initialize dictionary for the Q&A entry
                context = passage["text"]
                num_tokens = llm.get_num_tokens(context)
                if num_tokens>=120000:
                    print(f"Skipping {_id} num tokens: {num_tokens}") 
                    continue
                # Generate questions and answers with timing and error handling
                gen_start = time.time()
                try:
                    result = generate_questions_answers(context, client)
                    gen_time = time.time() - gen_start
                    total_gen_time += gen_time
                    gen_count += 1
                except Exception as e:
                    print(f"Error generating Q&A for {_id}: {e}")
                    data_count += 1
                    pbar.update(1)
                    continue
                
                # Extract questions and answers using regex
                questions: List[str] = re.findall(r"প্রশ্ন \d+: (.*?)(?:\n|$)", result)
                answers: List[str] = re.findall(r"উত্তর \d+: (.*?)(?:\n|$)", result)
                
                # Only proceed if at least 5 questions are generated
                if len(questions) >= 5:
                    entry["id"] = _id
                    entry["title"] = passage["meta"]["title"]
                    entry["text"] = context
                    entry["question"] = questions[:-1]
                    entry["answer"] = answers[:-1]
                    entry["num_entries"] = len(questions[:-1])
                    questions_count += len(questions[:-1])
                    #overall_count += len(questions[:-1])  # Update total questions
                    
                    # Save the entry as a JSON file
                    try:
                        with open(save_json, 'w', encoding='utf-8') as f:
                            json.dump(entry, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"Error saving {save_json}: {e}")
                        data_count += 1
                        pbar.update(1)
                        continue
                    
                    data_count += 1
            else:
                data_count += 1
                try:
                    with open(save_json, "r", encoding='utf-8') as f:
                        data = json.load(f)
                    questions_count += len(data["question"])
                except Exception as e:
                    print(f"Error reading existing {save_json}: {e}")
            
            # # Check if 60 seconds have passed since last update
            # current_time = time.time()
            # if current_time - last_update_time >= UPDATE_INTERVAL:
            #     overall_count = get_total_questions(save_dir)  # Recalculate total
            #     last_update_time = current_time
            #     print(f"Updated Total Qs after 60s: {overall_count}")
            
            # Calculate average generation time
            avg_gen_time = total_gen_time / gen_count if gen_count > 0 else 0
            
            # Update progress bar with metrics
            pbar.set_postfix({
                'Data': data_count,
                'Run Qs': questions_count,  # Questions in this run
                #'Total Qs': overall_count,  # Total questions across all runs
                'Avg Gen Time': f'{avg_gen_time:.2f}s'
            })
            pbar.update(1)


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Generate Q&A entries from JSON files with specified index"
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
    apis=pd.read_csv("/home/vpa/RAGollama3/data/chunks/apis.csv")
    api_key=apis["api"].tolist()[idx]
    
    # create client
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    )
    
    # Define and create the save directory
    save_dir: str = "/home/vpa/deliverables/qnagend2d3"
    
    # Generate Q&A for the filtered DataFrame
    gen_qna(df,save_dir,client)
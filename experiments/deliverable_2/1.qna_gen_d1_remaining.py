import pandas as pd
import json
from langchain_community.llms import Ollama
import os
import re
import time
import threading
import sys
from typing import List, Dict, Any

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

def show_spinner(stop_event: threading.Event) -> None:
    """
    Display a simple spinner in the console to indicate processing.

    Args:
        stop_event (threading.Event): Event to signal when to stop the spinner.
    """
    spinner = ['-', '\\', '|', '/']
    while not stop_event.is_set():
        for char in spinner:
            sys.stdout.write(f'\rProcessing... {char}')
            sys.stdout.flush()
            time.sleep(0.2)
    sys.stdout.write('\rProcessing complete!     \n')
    sys.stdout.flush()

def generate_questions_answers(context: str) -> str:
    """
    Generate questions and answers in Bengali based on the given context with a progress indicator.

    Args:
        context (str): The context to analyze and generate Q&A from.

    Returns:
        str: A string containing the generated questions and answers in the specified format.
    """
    prompt = template.format(context)
    
    # Set up threading to show progress spinner
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=show_spinner, args=(stop_event,))
    spinner_thread.start()
    
    # Generate Q&A using the LLM
    result = llm.invoke(prompt)
    
    # Stop the spinner once generation is complete
    stop_event.set()
    spinner_thread.join()
    
    return result

def gen_qna(df: pd.DataFrame) -> None:
    """
    Process a DataFrame of JSON file paths to generate Q&A entries and save them as JSON files.

    Args:
        df (pd.DataFrame): DataFrame containing a 'paths' column with JSON file paths.

    Notes:
        - Reads JSON files, generates Q&A using the context, and saves results in a specified directory.
        - Skips existing entries to avoid duplication.
    """
    data_count = 0  # Counter for processed entries
    
    # Iterate over each JSON file path in the DataFrame
    for json_path in df['paths'].tolist():
        with open(json_path, "r", encoding='utf-8') as f:
            passage = json.load(f)
        
        _id = passage["meta"]["id"]
        save_json = os.path.join(save_dir, f"{_id}.json")
        
        # Check if the Q&A file already exists
        if not os.path.exists(save_json):
            entry: Dict[str, Any] = {}  # Initialize dictionary for the Q&A entry
            context = passage["text"]
            
            # Generate questions and answers
            result = generate_questions_answers(context)
            
            # Extract questions and answers using regex
            questions: List[str] = re.findall(r"প্রশ্ন \d+: (.*?)(?:\n|$)", result)
            answers: List[str] = re.findall(r"উত্তর \d+: (.*?)(?:\n|$)", result)
            
            # Only proceed if at least 2 questions are generated
            if len(questions) >= 2:
                entry["id"] = _id
                entry["title"] = passage["meta"]["title"]
                entry["text"] = context
                entry["question"] = questions
                entry["answer"] = answers
                entry["num_entries"] = len(questions)
                
                # Save the entry as a JSON file
                with open(save_json, 'w', encoding='utf-8') as f:
                    json.dump(entry, f, ensure_ascii=False, indent=2)
                
                data_count += 1
                print(f"Entry created: Count: {data_count}")
        else:
            data_count += 1
            print(f"Entry Exists: Count: {data_count}")

if __name__ == "__main__":
    # Load passage and Q&A ID DataFrames
    d1_passage_df: pd.DataFrame = pd.read_csv("/home/vpa/deliverables/d1_ids.csv")
    d1_qna_df: pd.DataFrame = pd.read_csv("/home/vpa/deliverables/d1qna_ids.csv")
    
    # Filter out passages that already have Q&A entries
    df: pd.DataFrame = d1_passage_df.loc[~d1_passage_df['pid'].isin(d1_qna_df['pid'].tolist())]
    df.reset_index(inplace=True, drop=True)
    
    # Initialize the LLM model
    llm = Ollama(model="llama3.3", temperature=0.2, num_ctx=32768)
    
    # Define and create the save directory
    save_dir: str = "/home/vpa/deliverables/qnagend2d3"
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate Q&A for the filtered DataFrame
    gen_qna(df)
import os
import re
import json
from tqdm import tqdm
from glob import glob
def clean_text(raw_text):
    """
    Cleans the given raw Wikipedia article text.

    Parameters:
    - raw_text: The raw text extracted from the article

    Returns:
    - Cleaned text as a string
    """
    # Remove references like [1], [2], etc.
    text = re.sub(r"\[\d+\]", "", raw_text)
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Remove HTML entities
    text = re.sub(r"&[a-z]+;", "", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Preserve Bangla characters, spaces, and specific punctuation (.,?!)
    text = re.sub(r"[^\u0980-\u09FF\s.,()?!।]", "", text)
    
    return text

    
def remove_repeated_questions(questions):
    """
    Removes duplicate questions from the list and strips question numbers.

    Parameters:
    - questions: List of questions with numbers

    Returns:
    - A list of unique questions without numbers
    """
    cleaned_questions = set()
    unique_questions = []
    for q in questions:
        # Remove leading numbers and whitespace
        q_cleaned = re.sub(r"^\d+\.\s*", "", q)
        if q_cleaned not in cleaned_questions:
            cleaned_questions.add(q_cleaned)
            unique_questions.append(q_cleaned)
    return unique_questions




# Define your JSON directory
JSON_DIR = "wikiquestions"
JSON_DIR_SAVE="wikiquestions_cleaned"
total=0
# Iterate through chunks
for chunk_file in tqdm(glob(f"{JSON_DIR}/*.json")):
    with open(chunk_file, 'r', encoding='utf-8') as file:
        entry = json.load(file)
    
    
   
    wikiid = entry["id"]
    wikidump_text = entry["text"]
    # Clean the text
    cleaned_text = clean_text(wikidump_text)
    
    # Check if text has more than 50 words
    if len(cleaned_text.strip().split()) > 50:
        # Remove repeated questions
        unique_questions = remove_repeated_questions(entry.get("question", []))
        
        # Add the number of unique questions as a new key
        entry["num_entries"] = len(unique_questions)
        total+=len(unique_questions)
        # Update the questions in the entry
        entry["question"] = unique_questions
        
        # Update the text
        entry["text"] = cleaned_text
        
        
        # Save the processed data back to a new JSON file
        output_file =chunk_file.replace(JSON_DIR,JSON_DIR_SAVE)
        with open(output_file, 'w', encoding='utf-8') as out_file:
            json.dump(entry, out_file, ensure_ascii=False, indent=2)

print("Number of questions:",total)
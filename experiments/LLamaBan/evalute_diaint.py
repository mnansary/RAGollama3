import json
import random
from itertools import combinations
from Levenshtein import distance
from statistics import mean
from tqdm import tqdm
from langchain_community.llms import Ollama

# Provided prompt template and intent instruction
PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

Instruction:

{}

Input:

{}

Response:

{}"""

INTENT_INSTRUCTION = """
You are an AI assistant that extracts the intent from a given question or command.
The intent is represented as a hierarchical label in the format <category.subcategory[.sub-subcategory]>.
YOU WILL ONLY RESPOND WITH THE INTENT LABEL, WITHOUT ANY ADDITIONAL TEXT OR EXPLANATION.AND ONLY ONE INTENT LABEL.
The intent should be in English and should not contain any special characters or spaces.
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
"""

def extract_intent(llm, question: str) -> str:
    prompt = PROMPT_TEMPLATE.format(INTENT_INSTRUCTION, question, "")
    return llm(prompt).strip()

def levenshtein_similarity(s1, s2):
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - (distance(s1, s2) / max_len)

def compute_question_similarity(intents):
    pairs = list(combinations(intents, 2))
    similarities = [levenshtein_similarity(pair[0], pair[1]) for pair in pairs]
    return mean(similarities) if similarities else 0.0

def load_and_sample_questions(json_file_path, sample_size=100):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions = []
    for entry in data:
        if isinstance(entry, dict) and 'question' in entry:
            questions.extend(entry['question'])
    
    if len(questions) > sample_size:
        questions = random.sample(questions, sample_size)
    elif len(questions) < sample_size:
        print(f"Warning: Only {len(questions)} questions available, using all.")
    
    return questions

def generate_intents(model_name, questions, num_attempts=3):
    llm = Ollama(model=model_name, temperature=0.0, num_ctx=32768)
    intents = []
    for question in tqdm(questions, desc=f"Generating intents for {model_name}"):
        question_intents = [extract_intent(llm, question) for _ in range(num_attempts)]
        intents.append(question_intents)
    return intents

def evaluate_robustness(json_file_path, models=["llama3.3", "llama_d3"], sample_size=100):
    questions = load_and_sample_questions(json_file_path, sample_size)
    print(f"Processing {len(questions)} questions")
    
    results = {}
    for model_name in models:
        question_intents = generate_intents(model_name, questions)
        question_similarities = [compute_question_similarity(intents) for intents in question_intents]
        results[model_name] = mean(question_similarities) if question_similarities else 0.0
    
    # Print only overall average similarity
    for model_name, avg_similarity in results.items():
        print(f"Model: {model_name}, Overall Average Similarity: {avg_similarity:.4f}")
    
    return results

if __name__ == "__main__":
    json_file_path = "data/ministry_1k.json"
    evaluate_robustness(json_file_path)
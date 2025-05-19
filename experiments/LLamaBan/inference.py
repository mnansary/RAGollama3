import json
from tqdm import tqdm
from langchain_community.llms import Ollama

# Prompt templates
PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
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

QNA_INSTRUCTION = """
You are a knowledgeable chatbot. Answer the user's question in Bengali (বাংলা) using only the provided context.
If the context is insufficient or you’re unsure, respond exactly with 'NOT_SURE_ANSWER'.
"""

# Inference functions
def answer_question(llm, question: str, context: str) -> str:
    prompt = PROMPT_TEMPLATE.format(QNA_INSTRUCTION, context, question)
    return llm(prompt).strip()

def extract_intent(llm, question: str) -> str:
    prompt = PROMPT_TEMPLATE.format(INTENT_INSTRUCTION, question, "")
    return llm(prompt).strip()

# Main processing
def process_entries(entries, model_name, output_path):
    llm = Ollama(model=model_name, temperature=0.0, num_ctx=32768)
    output_data = []
    pbar = tqdm(total=len(entries), desc=f"Processing with {model_name}")

    for entry in entries:
        try:
            context = entry["text"]
            annotation_id = entry["annotation_id"]
            question = entry["question"][0].strip()  # Only first question
            answer = entry["answer"][0]  # Corresponding answer

            prediction = answer_question(llm, question, context)
            intent = extract_intent(llm, question)

            output_entry = {
                "context": context,
                "question": [question],
                "answer": [answer],
                "predictions": [prediction],
                "intents": [intent],
                "annotation_id": annotation_id
            }
            output_data.append(output_entry)
            pbar.update(1)

        except Exception as e:
            print(f"Error processing entry {entry.get('annotation_id', 'UNKNOWN')}: {e}")
            output_entry = {
                "context": context,
                "question": [question],
                "answer": [answer],
                "predictions": ["ERROR"],
                "intents": ["ERROR"],
                "annotation_id": annotation_id
            }
            output_data.append(output_entry)
            pbar.update(1)

    pbar.close()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Load and truncate to 100 entries
    input_path = "data/ministry_1k.json"
    with open(input_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    selected_data = full_data[:100]

    # Model 1
    process_entries(
        entries=selected_data,
        model_name="llama_d3",
        output_path="data/ministry_1k_llama_d3.json"
    )

    # Model 2
    process_entries(
        entries=selected_data,
        model_name="llama3.3",
        output_path="data/ministry_1k_llama3.3.json"
    )

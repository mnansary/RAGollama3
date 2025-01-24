#------------------------------------------
# imports
#------------------------------------------
import os
import time
from tqdm import  tqdm 
import json
from transformers import pipeline


#------------------------------------------
# dirs
#------------------------------------------

JSON_DIR="bangla_wikipedia_chunks"

#------------------------------------------
# globals
#------------------------------------------

pipe = pipeline(
    "text-generation",
    model="BanglaLLM/BanglaLLama-3-8b-BnWiki-Instruct",
    device="cuda"  # Explicitly use the GPU (device ID 0)
)
#------------------------------------------
# functions
#------------------------------------------
def format_text_into_contexts(wikidump_text, max_words=250):
    """
    Splits WikiDump text into contexts with complete paragraphs, ensuring no context exceeds max_words.

    Args:
        wikidump_text (str): The raw WikiDump text with paragraphs separated by '\n\n'.
        max_words (int): The maximum number of words allowed in each context.

    Returns:
        list: A list of text contexts.
    """
    paragraphs = wikidump_text.split("\n\n")  # Split text into paragraphs
    contexts = []  # To store formatted contexts
    current_context = []  # To build the current context
    current_word_count = 0  # Word count for the current context

    for paragraph in paragraphs:
        paragraph_word_count = len(paragraph.split())

        # Check if adding this paragraph would exceed the word limit
        if current_word_count + paragraph_word_count > max_words:
            # Save the current context and reset
            contexts.append("\n\n".join(current_context))
            current_context = []
            current_word_count = 0

        # Add the paragraph to the current context
        current_context.append(paragraph)
        current_word_count += paragraph_word_count

    # Add the last context if it has any content
    if current_context:
        contexts.append("\n\n".join(current_context))

    return contexts


def generate_bangla_questions(context, model, num_questions=11):
    """
    Generates unique Bangla questions from the given context with serial numbers.

    Args:
        context (str): The context text to generate questions from.
        model (object): The language model pipeline to process the input.
        num_questions (int): Number of unique questions to generate.

    Returns:
        list: A list of unique Bangla questions as strings with serial numbers.
    """
    # Define the prompt template with clear instructions
    prompt_template = f"""
    ### Instruction:
    নিচের প্রসঙ্গ থেকে {num_questions}টি অনন্য প্রশ্ন তৈরি করুন। 
    প্রতিটি প্রশ্ন নম্বর সহ আলাদা লাইনে লিখুন (যেমন: 1. প্রশ্ন, 2. প্রশ্ন ইত্যাদি)। 
    উত্তর অন্তর্ভুক্ত করবেন না, শুধুমাত্র প্রশ্ন তৈরি করবেন। 
    নিশ্চিত করুন যে একই প্রশ্ন বা অনুরূপ প্রশ্নের পুনরাবৃত্তি নেই।
    
    ### Input:
    {context}
    
    ### Response:
    """

    # Generate the output
    response = model(
        prompt_template,
        max_new_tokens=600 + len(context) // 4,  # Adjust max length based on context size
        num_return_sequences=1,
        do_sample=True,
        temperature=0.5,
        top_p=0.9  # Improves diversity while keeping coherence
    )

    # Extract and clean the generated text
    raw_output = response[0]['generated_text'].replace(prompt_template, "").strip()

    # Split into questions, cleaning each line
    questions = [q.strip() for q in raw_output.split("\n") if q.strip()]

    # Ensure the result contains no more than the requested number of questions
    return questions[:num_questions-1]


if __name__=="__main__":
    chunk_start=101
    chunk_stop=151
    for chunk in range(chunk_start,chunk_stop+1):
        chunk_file=os.path.join(JSON_DIR,f"chunk_{chunk:04d}.json")
        with open(chunk_file, 'r') as file:
            data = json.load(file)  
        for entry in tqdm(data):
            wikiid=entry["id"]
            wikidump_text=entry["text"]
            contexts=format_text_into_contexts(wikidump_text)
            entry["text"]=contexts[0]
            entry["question"]=generate_bangla_questions(contexts[0],pipe)
            output_file =  f"wikiquestions/{chunk}_{wikiid}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
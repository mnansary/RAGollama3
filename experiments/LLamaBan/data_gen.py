import json
import os
import random
from tqdm import tqdm
import requests

# --- Configuration ---
INSTRUCTION_TEXT = """You are a knowledgeable chatbot. Answer the user's question in Bengali (বাংলা) using only the provided context.
If the context is insufficient or you’re unsure, respond exactly with 'NOT_SURE_ANSWER'."""

# --- Helper Function ---
def get_token_count(passage):
    """
    Sends passage to the tokenization service and returns the token count.
    Returns -1 if the request fails.
    """
    try:
        response = requests.post(
            "http://0.0.0.0:3035/tokenize",
            json={"passage": passage},
            timeout=10
        )
        if response.status_code == 200:
            return response.json().get("tokens", -1)
        else:
            print(f"Tokenization service error: {response.status_code} - {response.text}")
            return -1
    except requests.RequestException as e:
        print(f"Error connecting to tokenization service: {e}")
        return -1

def load_and_process_json(file_path):
    """
    Loads a single JSON file and extracts all possible training examples.
    Each example is a dictionary: {"instruction": ..., "input": ..., "output": ...}
    Ignores the last question/answer pair.
    Only processes files where the 'text' field has fewer than 16,000 tokens.
    """
    examples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        context = data.get("text")
        questions = data.get("question")
        answers = data.get("answer")

        if not all([context, isinstance(questions, list), isinstance(answers, list)]):
            print(f"Warning: Missing 'text', 'question', or 'answer' in {file_path}, or they are not in expected format. Skipping.")
            return []

        # Check token count of the context
        token_count = get_token_count(context)
        if token_count == -1:
            print(f"Warning: Failed to get token count for {file_path}. Skipping.")
            return []
        if token_count >= 16000:
            print(f"Warning: Token count ({token_count}) for {file_path} is >= 16,000. Skipping.")
            return []

        if len(questions) != len(answers):
            print(f"Warning: Mismatch in length of questions ({len(questions)}) and answers ({len(answers)}) in {file_path}. Skipping.")
            return []

        if not questions:
            return []

        # Process all Q/A pairs except the last one
        num_pairs_to_process = len(questions) - 1

        for i in range(num_pairs_to_process):
            question = questions[i]
            answer = answers[i]

            # Format for Unsloth (Alpaca style)
            formatted_input = f"Context:\n{context}\n\nQuestion:\n{question}"

            examples.append({
                "instruction": INSTRUCTION_TEXT,
                "input": formatted_input,
                "output": answer
            })

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Skipping.")
    except Exception as e:
        print(f"An unexpected error occurred while processing {file_path}: {e}. Skipping.")
    return examples

def prepare_training_data(directories, num_required_samples):
    """
    Scans directories for JSON files, processes them, and returns a
    randomly selected subset of training examples.

    Args:
        directories (list): A list of directory paths to scan.
        num_required_samples (int): The desired number of training samples.

    Returns:
        list: A list of training example dictionaries, or an empty list if errors occur.
    """
    all_possible_examples = []
    for directory_path in directories:
        if not os.path.isdir(directory_path):
            print(f"Warning: Directory '{directory_path}' not found. Skipping.")
            continue

        print(f"Scanning directory: {directory_path}")
        for filename in tqdm(os.listdir(directory_path)):
            if filename.endswith(".json"):
                file_path = os.path.join(directory_path, filename)
                examples_from_file = load_and_process_json(file_path)
                all_possible_examples.extend(examples_from_file)

    if not all_possible_examples:
        print("No valid training examples could be generated.")
        return []

    print(f"\nTotal possible training examples generated: {len(all_possible_examples)}")

    if len(all_possible_examples) < num_required_samples:
        print(f"Warning: Requested {num_required_samples} samples, but only {len(all_possible_examples)} are available.")
        print("Using all available samples.")
        return all_possible_examples
    else:
        print(f"Randomly selecting {num_required_samples} samples.")
        return random.sample(all_possible_examples, num_required_samples)

# --- Main Execution ---
if __name__ == "__main__":
    DIRECTORIES_TO_SCAN = [
        "/home/vpa/deliverables/D2/Datasets/BanQA/data",
        "/home/vpa/deliverables/D3/Datasets/BanQA/data",
        "/home/vpa/deliverables/D4/Datasets/BanQA/data"
    ]
    NUM_TRAINING_SAMPLES_REQUIRED = 1000

    # --- Run the data preparation ---
    training_data = prepare_training_data(DIRECTORIES_TO_SCAN, NUM_TRAINING_SAMPLES_REQUIRED)

    if training_data:
        print(f"\nSuccessfully generated {len(training_data)} training samples.")
        
        # Optionally, save to a JSON file (often .jsonl for line-by-line JSON)
        output_filename = "data/prepared_training_data.jsonl"
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, "w", encoding="utf-8") as outfile:
            for entry in training_data:
                json.dump(entry, outfile, ensure_ascii=False)
                outfile.write("\n")
        print(f"\nTraining data saved to {output_filename}")
    else:
        print("\nNo training data was generated.")
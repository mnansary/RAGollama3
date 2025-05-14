import os
import json
import random
from typing import List, Dict, Tuple
from sentence_transformers import InputExample
from pathlib import Path
from sentence_transformers.util import cos_sim
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def get_similarity_score(query: str, passage: str, model: SentenceTransformer) -> float:
    """
    Compute cosine similarity between a query and a passage using the embedding model.
    Returns a float between -1 and 1.
    """
    query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
    passage_embedding = model.encode(passage, convert_to_tensor=True, show_progress_bar=False)
    
    # Compute cosine similarity
    similarity = cos_sim(query_embedding.unsqueeze(0), passage_embedding.unsqueeze(0)).item()
    return similarity

def generate_training_examples(
    directories: List[str],
    total_samples: int,
    save_dir: str,
    validation_split: float = 0.2,
    seed: int = 42,
    base_model: SentenceTransformer = None
) -> Tuple[List[InputExample], List[InputExample]]:
    """
    Generate training and validation examples for sentence transformers from JSON files.

    Args:
        directories (List[str]): List of directory paths containing JSON files.
        total_samples (int): Maximum total number of examples to generate (train + validation).
        save_dir (str): Directory path to save the generated datasets.
        validation_split (float, optional): Fraction of samples for validation (0 to 1). Defaults to 0.2.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        base_model (SentenceTransformer, optional): Base model to compute similarity scores for validation.

    Returns:
        Tuple[List[InputExample], List[InputExample]]: Training and validation InputExample lists.

    Notes:
        - JSON files must contain 'question', 'answer', and 'text' fields.
        - Each example consists of a query (randomly chosen from questions or answers) and its passage.
        - Training and validation datasets are saved as separate JSON files in save_dir.
        - Validation split is calculated as a fraction of total_samples.
        - If base_model is provided, validation examples are assigned similarity scores using get_similarity_score.
        - If more samples are needed than available files, the function cycles through files multiple times.
    """
    random.seed(seed)
    train_examples = []
    val_examples = []
    train_data = []
    val_data = []

    # Calculate number of validation samples
    num_val_samples = int(total_samples * validation_split)
    num_train_samples = total_samples - num_val_samples

    # Ensure save directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Collect all JSON files across directories
    all_files = []
    for dir_path in directories:
        if not os.path.exists(dir_path):
            continue
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".json")]
        print(f"Found {len(files)} files in directory: {dir_path}")
        all_files.extend(files)
    
    if not all_files:
        print("No JSON files found in the specified directories.")
        return [], []

    # Estimate iterations needed (cycle through files multiple times if necessary)
    files_per_cycle = len(all_files)
    estimated_cycles = max(1, (total_samples // files_per_cycle) + 1)
    total_iterations = files_per_cycle * estimated_cycles

    # Progress bar for processing files
    with tqdm(total=total_iterations, desc="Generating training samples") as pbar:
        iteration_count = 0
        while (len(train_examples) < num_train_samples or len(val_examples) < num_val_samples) and iteration_count < total_iterations:
            # Shuffle files for each cycle
            random.shuffle(all_files)
            for file_path in all_files:
                if len(train_examples) >= num_train_samples and len(val_examples) >= num_val_samples:
                    break
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data: Dict = json.load(f)
                    
                    questions = data.get("question", [])
                    answers = data.get("answer", [])
                    passage = data.get("text", "").strip()
                    
                    if len(questions) < 2 or len(answers) < 2 or not passage:
                        pbar.update(1)
                        iteration_count += 1
                        continue
                        
                    idx = random.randint(0, len(questions) - 2)
                    query = random.choice([questions[idx], answers[idx]]).strip()
                    
                    # Compute similarity score for validation examples if base_model is provided
                    score = get_similarity_score(query, passage, base_model) if base_model and len(val_examples) < num_val_samples else 1.0
                    
                    # Create InputExample and store data
                    example = InputExample(texts=[query, passage], label=score)
                    example_data = {
                        "query": query,
                        "passage": passage,
                        "score": score,
                        "source_file": os.path.basename(file_path),
                        "source_directory": os.path.dirname(file_path)
                    }
                    
                    # Assign to train or validation split
                    if len(train_examples) < num_train_samples:
                        train_examples.append(example)
                        train_data.append(example_data)
                    elif len(val_examples) < num_val_samples:
                        val_examples.append(example)
                        val_data.append(example_data)
                    
                    pbar.update(1)
                    iteration_count += 1
                    
                except Exception as e:
                    pbar.update(1)
                    iteration_count += 1
                    continue

    # Print the number of generated samples
    print(f"Generated {len(train_examples)} training samples and {len(val_examples)} validation samples.")

    # Save the generated datasets
    if train_data:
        train_save_path = os.path.join(save_dir, "training_examples.json")
        with open(train_save_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    if val_data:
        val_save_path = os.path.join(save_dir, "validation_examples.json")
        with open(val_save_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)

    return train_examples, val_examples
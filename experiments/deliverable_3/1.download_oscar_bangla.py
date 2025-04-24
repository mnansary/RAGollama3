from datasets import load_dataset
import pandas as pd
from tqdm.auto import tqdm
import os
import time

# Language pattern
languages = {"bn": u'[\u0980-\u09FF]+'}

def download_with_retry(dataset_name, config, retries=200000, sleep_secs=10):
    for attempt in range(retries):
        try:
            return load_dataset(dataset_name, config)
        except Exception as e:
            print(f"Download failed (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                print(f"Retrying in {sleep_secs} seconds...")
                time.sleep(sleep_secs)
            else:
                raise

def main():
    data_dir = "/home/vpa/deliverables/__archive__/oscar"
    os.makedirs(data_dir, exist_ok=True)
    max_len = 50000
    
    for lang in languages:
        print(f"Downloading dataset for: {lang}")
        dataset = download_with_retry("oscar", f"unshuffled_deduplicated_{lang}")

        print(f"Chunking and saving to CSVs...")
        for idx in tqdm(range(0, dataset["train"].num_rows, max_len)):
            try:
                chunk = dataset["train"]["text"][idx:idx + max_len]
                df = pd.DataFrame({"text": chunk})
                df.to_csv(os.path.join(data_dir, f"{lang}_{idx}.csv"), index=False)
            except Exception as e:
                print(f"Error processing chunk {idx}: {e}")
                continue
        print(f"Finished downloading and processing dataset for: {lang}")

if __name__ == "__main__":
    main()

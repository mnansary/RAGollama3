"""
Script for deliverable Banserv2sql
"""

import mwxml
import json
import mwparserfromhell
from tqdm import tqdm
import os
import re
import bz2


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


    
def process_dump_in_chunks(dump_path, output_dir):
    """
    Process the Wikipedia dump and save articles in chunks of JSON files.
    
    Parameters:
    - dump_path: Path to the .bz2 dump file
    - output_dir: Directory to save the resulting JSON files
    - chunk_size: Number of articles per JSON file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dump = mwxml.Dump.from_file(open(dump_path, 'rb'))
    articles = []
    data_count = 0

    print("Processing articles...")
    for page in tqdm(dump.pages):
        if not page.redirect and page.namespace == 0:  # Ignore redirects and non-articles
            for revision in page:
                try:
                    # Parse the article content
                    wikicode = mwparserfromhell.parse(revision.text or "")
                    text = wikicode.strip_code()

                    # Clean the text
                    cleaned_text = clean_text(text)
                    if len(cleaned_text.split())>100:
                        # Save article data
                        article = {
                            "meta":{"title": page.title,
                                    "id": page.id,
                                    "source":"wikidump"},
                            "text": cleaned_text
                        }
                        output_file = os.path.join(output_dir, f"{data_count}.json")    
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(article, f, ensure_ascii=False, indent=2)
                        data_count+=1
                except Exception as e:
                    print(f"Error processing page {page.title}: {e}")
            if data_count >= required_data:
                print("Processing complete!")
                break


if __name__=="__main__":
    # Define the file paths
    input_file = "bnwiki-dump.xml.bz2"
    output_file = "bnwiki-dump.xml"

    # Decompress the file
    with bz2.BZ2File(input_file, 'rb') as bz2_file:
        with open(output_file, 'wb') as xml_file:
            for data in iter(lambda: bz2_file.read(100 * 1024), b''):  # Read in chunks
                print("processing data")
                xml_file.write(data)

    save_dir="/home/vpa/deliverables/BanSERV2SQL/passages/"
    required_data=60000
    dump_file = "bnwiki-dump.xml"
    process_dump_in_chunks(dump_file, save_dir)

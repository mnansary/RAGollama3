import os 
from glob import glob
import pandas as pd 
import json 
from tqdm import tqdm
tqdm.pandas()
import re 

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

if __name__=="__main__":
    data_dir = "/home/vpa/deliverables/__archive__/oscar"
    save_dir = "/home/vpa/deliverables/__archive__/oscar_passages"
    os.makedirs(save_dir,exist_ok=True)
    csvs=[_c for _c in glob(os.path.join(data_dir,"*.csv"))]
    dfs=[]
    for csv in tqdm(csvs):
        df=pd.read_csv(csv)
        dfs.append(df)
    df=pd.concat(dfs,ignore_index=True)
    df.to_csv(os.path.join(data_dir,"oscar_bangla.csv"),index=False)
    for idx in tqdm(range(len(df))):
        text=df.loc[idx,"text"]
        
        text=clean_text(text)
        if len(text.split())>100:
            data={}
            data["meta"]={}
            data["meta"]["id"]=idx
            data["meta"]["title"]=f"passage_{idx}"
            data["meta"]["source"]="oscar"
            data["text"]=text
            output_file = os.path.join(save_dir, f"oscar_{idx}.json")    
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
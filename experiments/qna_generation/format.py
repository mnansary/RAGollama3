



import os
import json
import pandas as pd 
from tqdm import  tqdm 
from glob import glob
tqdm.pandas()

#------------------------------------------
# dirs
#------------------------------------------

DATA_DIR="questions"
SAVE_DIR="questions_with_metadata"
DATA_CSV="merged_processed-passages-20250101.csv"

#------------------------------------------
# functions
#------------------------------------------

dfin=pd.read_csv(DATA_CSV)
dfin = dfin[dfin['text'].notna()]
dfin=dfin[["text",'site_name','passage_heading']]
dfin.dropna(inplace=True)
df=pd.read_csv(DATA_CSV)
df=df.loc[df.index.isin(dfin.index)]

for idx, row in tqdm(df.iterrows(),total=len(df)):
    row_json = row.to_json(force_ascii=False)
    row_dict = json.loads(row_json)
    row_dict.pop('text')
    row_dict["topic"] = row_dict.pop('sub_category/topic')
    row_dict["translated"] = row_dict.pop('translated(add "True" if translated)')
    row_dict["update_date"] = row_dict.pop('website_last_updated_at (yyyy-mm-dd)')
    jsons=[f for f in glob(os.path.join(DATA_DIR,f"{idx}_*.json")) if "wrong" not in f]
    for _json in jsons:
        with open(_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.pop('response')
        data.update(row_dict)
        output_file =  _json.replace(DATA_DIR,SAVE_DIR)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

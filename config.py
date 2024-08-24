# path to the created database
VECTOR_DB_PATH="data/database"

# The path of the csv file that will be used to create a vectorstore. 
# # The csv will have only one column and each row entry will be treated as a separate document.  
DATA_SOURCE="data/source.csv" 

# The LLM model to be used for RAG
MODEL_NAME="llama3.1:70b"       

# The model serving url
MODEL_BASE_URL='http://localhost:11434'  

# The model to use in sentence-transformers for creation embedding
EMBEDDING_MODEL="l3cube-pune/bengali-sentence-bert-nli" 
# path to the created database
VECTOR_DB_PATH="prototype/database"

# The path of the csv file that will be used to create a vectorstore. 
# # The csv will have only one column and each row entry will be treated as a separate document.  
DATA_SOURCE="prototype/source.csv" 

# The LLM model to be used for RAG
MODEL_NAME="llama3.1:8b"       

# The model to use in sentence-transformers for creation embedding
EMBEDDING_MODEL="intfloat/multilingual-e5-large"
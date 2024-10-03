# RAGollama3
RAG implementation with ollama 3.1 70B

# Setup instructions


## Ollama Setup
* install ollama following the instructions from [official site](https://ollama.com/download/linux) according to your OS

    * for linux : ```curl -fsSL https://ollama.com/install.sh | sh```

* open a terminal and serve ollama

```bash
ollama serve
```
* open a terminal and run ollama3.1: 70B model and keep it running 

```bash
ollama run llama3.1:70b
```

> close the terminal when you want to shut down the application. Whenever you want to run the application , you have repeat this step of running ollama3.1.


## Python environment Setup
* create a python environment 

```bash
conda create -n ragollama python=3.10
```

* activate your environment

```bash
conda activate ragollama
```

* install dependencies

```bash
pip install -r requirements.txt
```


* set the following values in **config.py** properly

```python
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
EMBEDDING_MODEL="l3cube-pune/bengali-sentence-similarity-sbert" 
```


* create the vector store

```bash 
python create_vectorstore.py
```

* run the streamlit application

> The following command runs the app on port 3035. Use different port if needed.   

```bash
streamlit run app.py --server.port 3035
```
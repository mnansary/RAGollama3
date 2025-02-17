from flask import Flask, request, jsonify
from langchain_community.llms import Ollama

app = Flask(__name__)
llm = Ollama(model="llama3.3", num_ctx=32768)

@app.route('/tokenize', methods=['POST'])
def tokenize():
    data = request.json
    passage = data.get("passage", "")
    if not passage:
        return jsonify({"error": "No passage provided"}), 400
    
    num_tokens = llm.get_num_tokens(passage)
    return jsonify({"tokens": num_tokens})

if __name__ == '__main__':
    app.run(debug=False,port=3035,host="0.0.0.0")
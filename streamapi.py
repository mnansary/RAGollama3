from flask import Flask, request, Response, jsonify
from model import BanglaRAGService  # Assuming your BanglaRAGService is in bangla_rag_service.py
import traceback

app = Flask(__name__)
rag_service = BanglaRAGService() # Initialize your RAG service


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "Question is required"}), 400

    def stream_response():
        try:
            for chunk in rag_service.process_query(question):
                yield f"{chunk}\n"  # Ensure new lines for streaming clients
                import sys
                sys.stdout.flush()  # Force immediate flush of output
        except Exception as e:
            error_message = f"Backend Error: {str(e)}"
            traceback.print_exc()
            yield error_message + "\n"

    return Response(stream_response(), mimetype='text/plain') # Or 'text/plain' if you don't want SSE


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=3030,threaded=True) # Run Flask app
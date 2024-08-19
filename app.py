from flask import Flask, request, jsonify
from query_data_pinecone import get_answer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/get_answer',methods = ['POST'])
def get_answer_route():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({"error": "Question parameter is required"}), 400
    
    try:
        answer = get_answer(question)
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        return jsonify({"error":str(e)}),500


if __name__ == "__main__":
    app.run()

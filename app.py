from flask import Flask, render_template, request
from rag import execute_rag
# import jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process_question', methods=['POST'])
def get_result():
    if request.method == 'POST':
        question = request.form['user_input']
        answer = execute_rag(question)
        return render_template("index.html",  answer=answer)
    else:
        return "error"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
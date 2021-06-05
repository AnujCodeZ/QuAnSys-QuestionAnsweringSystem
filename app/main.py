from flask import Flask, render_template, request, redirect, url_for
from app.infer import predict

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        question = request.form['question']
        context = request.form['context']
        results = predict(context, question)
        answer_before = results['answer_before']
        answer = results['answer']
        answer_after = results['answer_after']
        return answer_page(answer_before=answer_before,
                    answer=answer, answer_after=answer_after)
    else:
        return render_template('index.html')

@app.route('/answer_page', methods=['GET', 'POST'])
def answer_page(answer_before, answer, answer_after):
    return render_template('answer.html', answer_before=answer_before,
                    answer=answer, answer_after=answer_after)

if __name__ == '__main__':
    app.run(debug=True)
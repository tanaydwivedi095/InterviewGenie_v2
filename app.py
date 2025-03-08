from flask import Flask, render_template, request, redirect, url_for

from src.helper.interviewee import get_answer
from src.helper.interviewer import Interviewer

app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        username = request.form.get('username')
        model_type = request.form.get('model_type')
        if model_type == 'interviewee':
            return redirect(url_for("interviewee", username=username))  # Correct redirect
        else:
            return redirect(url_for("interviewer", username=username))  # Correct redirect


@app.route('/interviewer', methods=['GET', 'POST'])
def interviewer():
    if request.method == 'GET':
        username = request.args.get('username', 'Guest')  # Default to "Guest" if missing
        return render_template("interviewer.html", username=username)

    elif request.method == 'POST':
        username = request.form.get('username', 'Guest')  # Get username from form
        model_type = request.form.get('model_type')

        if model_type == "free":
            print("RUNNING THE FREE MODEL")
            interviewer_obj = Interviewer(username, "Free")
            questions = interviewer_obj.generate_questions()
            questions = questions[1::2]
            questions = "#".join(questions)
            return redirect(url_for("question_sheet", username=username, questions=questions, model_type=model_type))

        elif model_type == "paid":
            print("RUNNING THE PAID MODEL")
            api_key = request.form.get('api_key')
            difficulty = request.form.get('difficulty')
            interviewer_obj = Interviewer(username, "Paid", api_key, difficulty)
            questions = interviewer_obj.generate_questions()
            questions = "#".join(questions)
            return redirect(url_for("question_sheet", questions=questions, username=username, model_type=model_type, api_key=api_key, difficulty=difficulty))

@app.route('/question_sheet', methods=['GET', 'POST'])
def question_sheet():
    questions = request.args.get('questions')
    username = request.args.get('username')
    model_type = request.args.get('model_type')
    api_key = request.args.get('api_key')
    difficulty = request.args.get('difficulty')
    print(f"model type in question sheet : {model_type}")
    return render_template("question_sheet.html", questions=questions, username=username, model_type=model_type, api_key=api_key, difficulty=difficulty)

@app.route('/interviewee', methods=['GET', 'POST'])
def interviewee():
    username = request.args.get('username')
    model_type = request.args.get('model_type')
    if request.method == 'GET':
        print(f"model type in interviewee GET : {model_type}")
        return render_template("interviewee.html", username=username, model_type=model_type)
    elif request.method == 'POST':
        query = request.form.get('query')
        answer = get_answer(query)
        print(f"model type in interviewee POST : {model_type}")
        return render_template("interviewee.html", username=username, answer=answer, model_type=model_type)

@app.route('/submit_answer', methods=['POST'])
def submit_answers():
    username = request.form.get('username', 'Guest')
    model_type = request.form.get('model_type')
    api_key = request.form.get('api_key')
    difficulty = request.form.get('difficulty')
    questions = 5
    answers = []
    questionbank = []
    for i in range(questions):
        answer_key = f'answer_{i}'
        question_key = f"question_{i}"
        question = request.form.get(question_key)
        questionbank.append(question)
        answer = request.form.get(answer_key)
        answers.append(answer)
    filtered_questions = []
    for question in questionbank:
        if question!=None:
            filtered_questions.append(question)
    filtered_answers = []
    for answer in answers:
        if answer!=None:
            filtered_answers.append(answer)
    questions = "#".join(filtered_questions)
    answers = "#".join(filtered_answers)
    print(f"model type in submit answers : {model_type}")
    return redirect(url_for("result", questions=questions, answers=answers, username=username, model_type=model_type, api_key=api_key, difficulty=difficulty))

@app.route('/result', methods=['GET', 'POST'])
def result():
    questions = request.args.get('questions').split('#')
    answers = request.args.get('answers').split('#')
    username = request.args.get('username')
    model_type = request.args.get('model_type')
    api_key = request.args.get('api_key')
    difficulty = request.args.get('difficulty')
    score = list()
    if model_type == "paid":
        interviewer_obj = Interviewer(username, "Paid", api_key, difficulty)
        for i in range(0, len(questions)):
            question = questions[i]
            answer = answers[i]
            score.append(interviewer_obj.score_answer(question, answer))
    print(f"model type in result : {model_type}")
    return render_template("result.html", questions=questions, answers=answers, username=username, model_type=model_type, api_key=api_key, difficulty=difficulty, score=score)


if __name__ == '__main__':
    app.run(
        host='127.0.0.1',
        port=8080,
        debug=True
    )

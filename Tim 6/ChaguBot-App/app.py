from process import preparation, botResponse
from flask import Flask, render_template, request, jsonify

# download nltk
preparation()

app = Flask(__name__)
app.static_folder = 'static'


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Team')
def team():
    return render_template('team.html')


@app.route('/Predict', methods=["GET", "POST"])
def predict():
    text = request.get_json().get("message")
    response = botResponse(text)
    message = {"answer": response}
    return jsonify(message)


if __name__ == '__main__':
    app.run(debug=True)
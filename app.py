import pickle
import sklearn
from sklearn.ensemble import RandomForestRegressor
import flask
from flask import render_template

app = flask.Flask(__name__, template_folder='template')

@app.route('/', methods=['POST', 'GET'])

@app.route('/index', methods=['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')

    if flask.request.method == 'POST':
        with open('model_rfr1.pkl', 'rb') as f:
            loaded_model1 = pickle.load(f)
        with open('model_rfr2.pkl', 'rb') as b:
            loaded_model2 = pickle.load(b)
        x1 = float(flask.request.form['IW'])
        x2 = float(flask.request.form['IF'])
        x3 = float(flask.request.form['VW'])
        x4 = float(flask.request.form['FP'])
        y_rfr1 = loaded_model1.predict([[x1, x2, x3, x4]])
        y_rfr2 = loaded_model2.predict([[x1, x2, x3, x4]])
        return render_template('main.html', result=(y_rfr1, y_rfr2))


if __name__ == '__main__':
    app.run()

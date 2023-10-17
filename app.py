import flask
from flask import render_template
import pickle
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
app = flask.Flask(__name__, template_folder= 'templates')
@app.route('/', methods = ['POST', 'GET'])
@app.route('/index', methods = ['POST', 'GET'])

def main ():
     if flask.request.method == 'GET':
         return render_template ('main.html')
     if flask.request.method == 'POST':
         with open ('model_gbr1.pkl', 'rb') as f:
             loaded_model = pickle.load (f)
         with open ('model_gbr2.pkl', 'rb') as b:
             loaded_model2 = pickle.load (b)
         X1 = float (flask.request.form['IW'])
         X2 = float (flask.request.form['IF'])
         X3 = float (flask.request.form['VW'])
         X4 = float (flask.request.form['FP'])
         y1_gbr1 = loaded_model.predict([[X1,X2,X3,X4]])
         y2_gbr2 = loaded_model2.predict([[X1,X2,X3,X4]])

         return render_template('main.html', result = (y1_gbr1, y2_gbr2) )

if __name__ == '__main__':
     app.run()
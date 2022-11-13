from ctypes.wintypes import FLOAT
from flask import Flask, json, jsonify, render_template, request, url_for
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#Initialize the flask App
import pickle 
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

#---------------------------------------------------------------------------------
@app.route("/prediction")
def pred():
    return render_template('prediction.html')

@app.route('/predictionR', methods=['POST','GET'])
def predictionR():
    if request.method == 'POST':
        c1 = request.form['C1']
        c2 = request.form['C2']
        c3 = request.form['C3']
        c4 = request.form['C4']
        c5 = request.form['C5']
        c6 = request.form['C6']
        c7 = request.form['C7']
        c8 = request.form['C8']
        c9 = request.form['C9']
        c10 = request.form['C10']
        c11 = request.form['C11']
        c12 = request.form['C12']
        c13 = request.form['C13']   
    #input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)
    input_data = (float(c1),float(c2),float(c3),float(c4),float(c5),float(c6),float(c7),float(c8),float(c9),float(c10),float(c11),float(c12),float(c13))
    # change the input data to a numpy array
    input_data_as_numpy_array= np.asarray(input_data)
    # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    value_predicted = model.predict(input_data_reshaped)
    if (value_predicted[0]== 0):
        res = 'The Person does not have a Heart Disease'
    else:
        res = 'The Person has Heart Disease'

    return render_template("predR.html",prediction_text=res)
app.jinja_env.globals.update(predictionR=predictionR)  

if __name__ == "__main__":
    app.run()
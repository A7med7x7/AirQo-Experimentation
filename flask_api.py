from flask import request,app,Flask
import pickle
#from lightgbm import LGBMRegressor
import pandas as pd 
import numpy as np 
import lightgbm 

app = Flask(__name__)
pickle_in = open('/Users/ahmed/Downloads/computerscience/AirQo/PM2.5-Prediction/model/pm2_5.pkl','rb')
regressor = pickle.load(pickle_in)

@app.route('/the-server')
def welcome():
    return('welcome to the home directory')


@app.route('/predict',method=['POST'])
def prediction_note():
    df_test = pd.read_csv(request.files.get('file'))
    prediction = regressor.predicted
    return f'The predicted value is: {prediction[0]}'

if __name__ == '__main__':
    app.run()
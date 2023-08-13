import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

def train_test_split(data):
    # data = data.set_index([data.columns[0]])
    data_size = len(data)
    train = data[:-int((0.25 * data_size))]
    test = data[-int((0.25 * data_size)):]
    return data, train, test

def auto_reg_modelling(train, test, train_exog=None, test_exog=None):                
    model = AutoReg(train, lags=12, exog=train_exog).fit()
    prediction = model.predict(len(train), len(train)+len(test)-1, exog_oos=test_exog)
    prediction = pd.DataFrame({'data':prediction})
    
    return prediction

def evaluate_model(test, prediction):
    mape = mean_absolute_percentage_error (test, prediction) * 100
    accuracy = 100 - mape
    mae = mean_absolute_error(test, prediction)
    mse = mean_squared_error(test, prediction)
    rmse = np.sqrt(mean_squared_error(test, prediction))
    return round(accuracy,2), round(mae,2), round(mse,2), round(rmse,2)
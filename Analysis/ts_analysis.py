__author__ = 'Lei Huang'

from scipy import signal
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
import numpy as np

def detrend(data):
    detrended = signal.detrend(data)
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(data)
    axes[1].plot(detrended)
    plt.show()

def ar(data):
    model = AutoReg(data, lags=1)
    model_fit = model.fit()
    yhat = model_fit.predict(len(data), len(data))
    print('yhat:', yhat)
    return yhat

def arima(data):
    # ARIMA example
    # fit model
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(len(data), len(data), typ='levels')
    print(yhat)
    return yhat

def ma(data):
    model = ARMA(data, order=(0, 1))
    model_fit = model.fit(disp=False)
    # make prediction
    yhat = model_fit.predict(0, len(data))
    print(yhat)
    return yhat

def convolve_sma(array, period):
    ma = np.convolve(array, np.ones((period,))/period, mode='valid')
    ma = np.insert(ma, 0, np.zeros(period-1) )
    #plt.plot(array)
    #plt.plot(ma)

    #print(array.size, ma.size)
    #plt.show()
    #print(ma)
    return ma
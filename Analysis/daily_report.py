__author__ = 'Lei Huang'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from fit import *
import torch
import math
import csv
import locale
import datetime
import os
from dask import delayed
from ts_analysis import *
from infect_model import *
from plot_data import *
from glob import glob
from dateutil.parser import parse

def loadDailyReports():
    filenames = sorted(glob('../Data/COVID-19/csse_covid_19_data/csse_covid_19_daily_reports_us/*.csv'))
    Incident_rate={}
    Testing_rate={}
    Confirmed = {}
    Tested = {}
    df = pd.read_csv(filenames[0])
    df = df[(df['Country_Region'] == "US")]
    for index, row in df.iterrows():
        if pd.isnull(row['People_Tested']):
            continue
        Incident_rate[row['Province_State']] =np.array([], dtype=np.float)
        Testing_rate[row['Province_State']] = np.array([], dtype=np.float)
        Confirmed[row['Province_State']] = np.array([], dtype=np.int64)
        Tested[row['Province_State']] = np.array([], dtype=np.int64)

    dates=[]
    for file in filenames:
        dates.append(parse(os.path.splitext(os.path.basename(file))[0]))
        df = pd.read_csv(file)
        df = df[(df['Country_Region'] == "US")]
        for index, row in df.iterrows():
            if pd.isnull(row['People_Tested']):
                continue
            Incident_rate[row['Province_State']] = np.append(Incident_rate[row['Province_State']], row['Incident_Rate'])
            Testing_rate[row['Province_State']] = np.append(Testing_rate[row['Province_State']],row['Testing_Rate'])
            Confirmed[row['Province_State']] = np.append( Confirmed[row['Province_State']], row['Confirmed'])
            Tested[row['Province_State']] = np.append(Tested[row['Province_State']], row['People_Tested'])

    print(Incident_rate['New York'])
    print(Testing_rate['New York'])
    print(Confirmed['New York'])
    print(Tested['New York'])
    return dates, Incident_rate, Testing_rate, Confirmed, Tested


def daily_process():
    os.makedirs('Results', exist_ok=True)
    dates, Incident_rate, Testing_rate, Confirmed, Tested = loadDailyReports()

    #C = np.insert(Confirmed['New York'], 0, 0)
    C = np.diff(Confirmed['New York'])

    #T = np.insert(Tested['New York'], 0, 0)
    T = np.diff(Tested['New York'])

    SC = sum(Confirmed.values())
    SC = np.diff(SC)

    ST = sum(Tested.values())
    print(ST)
    ST = np.diff(ST)
    print(ST)

    months = mdates.MonthLocator()  # every month
    days =  mdates.DayLocator()
    months_fmt = mdates.DateFormatter('%m/%d')


    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.xaxis.set_major_locator(months)
    ax1.xaxis.set_major_formatter(months_fmt)
    ax1.xaxis.set_minor_locator(days)
    ax1.plot_date(dates[1:], ST, 'b-', label='Tested')
    ax1.set_ylabel('Tested Cases')
    #ax1.set_yscale('log')
    #plt.plot(SC, label='Confirmed')
    #plt.plot(ST, label='Tested')
    ax1.legend(loc='upper left')
    ax1.set_ylim(1.0e+4, 3.5e+5)
    ax1.set_xlim(datetime.datetime(2020,4,10), datetime.datetime(2020,4,30))
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    #plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)


    ax12 = ax1.twinx()
    ax12.xaxis.set_major_locator(months)
    ax12.xaxis.set_major_formatter(months_fmt)
    ax12.xaxis.set_minor_locator(days)
    #ax2.plot_date(dates[1:], (SC/ST).astype(float), 'r-')
    ax12.plot_date(dates[1:], SC, 'r-', label='Confirmed')
    ax12.set_ylabel('Confirmed Cases', color='r')
    ax12.set_ylim(2.0e+4, 5.0e+4)
    for tl in ax12.get_yticklabels():
        tl.set_color('r')
    ax12.legend(loc='upper right')
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax1.grid(True, which="both", linestyle='--')
    ax12.grid(True, linestyle='-')

    ax2.xaxis.set_major_locator(months)
    ax2.xaxis.set_major_formatter(months_fmt)
    ax2.xaxis.set_minor_locator(days)
    ax2.set_xlim(datetime.datetime(2020,4,10), datetime.datetime(2020,4,30))
    ax2.plot_date(dates[1:], (SC/ST).astype(float), 'g-', label='Positive Rate')
    ax2.legend()
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax2.grid(True, which="both", linestyle='--')
    plt.title("US Testing Data: " + dates[-1].strftime("%m/%d/%Y"))
    plt.savefig('Results/US_Testing_Daily.png')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    daily_process()


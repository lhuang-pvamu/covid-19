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

pd.set_option('display.max_columns', 500)
locale.setlocale(locale.LC_ALL, 'en_US')

def process(world, country, type):

    config = dict()
    length = 130
    config['x_limits'] = [0, length]
    config['nx'] = length+1
    config['dx'] = (config['x_limits'][1] - config['x_limits'][0]) / (config['nx'] - 1)

    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    days =  mdates.DayLocator()
    months_fmt = mdates.DateFormatter('%m/%d')

    os.makedirs('Results', exist_ok=True)
    os.makedirs('Figures', exist_ok=True)

    #NY = gauss_model(config, 34.38, 47588, 39.03)
    #print(NY)

    #US = gauss_model(config, 36, 97000, 64)
    #print(US)

    #df = pd.read_csv("../csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv")
    if world:  # global data
        #if type == "confirmed":
        df_c = pd.read_csv("../Data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
        #else:
        df_d = pd.read_csv("../Data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
        column_names = pd.to_datetime(df_c.columns[4:], format='%m/%d/%y')
    else:      # US data
        if type == "confirmed":
            offset = 0
        else:
            offset=1
        df_c = pd.read_csv("../Data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv")
        df_d = pd.read_csv("../Data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv")

        column_names = pd.to_datetime(df_c.columns[11+offset:], format='%m/%d/%y')

    rows, columns = df_c.shape
    dates = np.array([column_names[0] + np.timedelta64(i, 'D')
                      for i in range(length+1)])


    print(column_names, column_names.size)
    print(dates)
    print(rows, columns)
    #df.columns=df.columns.str.replace('/','_')

    #df = df[(df['Country/Region'] == 'US') & (df['Province/State'] == 'New York')]
    #df = df[(df['Country/Region'] == 'US') & (df['Province/State'] == 'Texas')]
    if world:
        if isinstance(country, list):
            countryList = country
        elif country == 'ALL':
            countryList = df_c['Country/Region'].unique()
        else:
            countryList = [country]
        keyword = 'Country/Region'
        startIndex = 2
    else:
        countryList = df_c['Province_State'].unique()
        countryList = np.append(countryList, "US ACCUM")
        keyword = 'Province_State'
        startIndex = 5+offset

    import matplotlib.cbook as cbook
    new_column_names = [keyword]
    for date in dates:
        new_column_names.append(date.strftime('%m-%d'))
    new_column_names.append('Total')
    out = csv.writer(open("Results/prediction.csv", "w"), delimiter=',')
    #out.write('Country')
    out.writerow(new_column_names)
    totalData = np.zeros(column_names.size)
    totalPred = np.zeros(dates.size)

    if type == "confirmed":
        predicted_label = 'Predicted Daily New Cases'
        data_label = 'Confirmed Daily New Cases'
    else:
        predicted_label = 'Predicted Daily Deaths'
        data_label = 'Confirmed Daily Deaths'


    futures = []

    for c in countryList:
        print("=============== ", c, " =============")
        if c == 'Recovered':
            print('--- Skip ---')
            continue

        if c != 'US ACCUM':
            df_c_c = df_c[(df_c[keyword] == c)]
            df_c_d = df_d[(df_d[keyword] == c)]
            #for v in df_c.values:
            #    print(v)
            df_c_c = df_c_c.groupby(keyword).sum()
            df_c_d = df_c_d.groupby(keyword).sum()
            #print(df_c.values)
            #print(df_d.values)
            #print(df['Country/Region'].sum())
            #df = df[df.Country_Region=='US']
            #print(df_c.values[0])


            data_c = np.insert(df_c_c.values[0][startIndex:].astype(float), 0, 0)
            data_d = np.insert(df_c_d.values[0][startIndex:].astype(float), 0, 0)
            print(data_c, data_c.size)
            print(data_d, data_d.size)

            #print(len(data))
            data_c = np.diff(data_c)
            data_d = np.diff(data_d)
            #detrend(data)
            #ar(data)
            #arima(data)
            #ma(data)
            ma = convolve_sma(data_c, 5)

            data2 = np.insert(data_c, 0, 0)
            data2 = np.diff(data2)
            print(data2)
            datemin = np.datetime64(column_names[0])
            datemax = datemin + np.timedelta64(length, 'D')
            #print(datemin, datemax)

            #print(torch.from_numpy(data).size())
            curve,position,amp,span = fit(torch.from_numpy(data_c).double(), config, dist='gaussian')
            #futures.append([curve,position,amp,span])
            if (math.isnan(position) or math.isnan(amp) or math.isnan(span)):
                continue

            totalData += data_c
            totalPred += curve

            if (position > curve.size):
                print(c, " position is out of bound: ", position)
                position = curve.size-1

            print(position, amp, span)

            max_val = max(curve.max(), data_c.max())
        else:
            max_val = max(totalPred.max(), totalData.max())
            ma = convolve_sma(totalData, 5)


        fig, ax = plt.subplots()
        #ax.xaxis.set_major_locator(months)
        #ax.xaxis.set_major_formatter(months_fmt)
        #ax.xaxis.set_minor_locator(days)
        ax.set_xlim(datemin, datemax)

        if c!='US ACCUM':
            #ax.bar(column_names, data, width=1, color='orange')
            plt.fill_between(column_names, y1=data_c, y2=0, alpha=0.5, linewidth=2, color='orange')
            plt.plot_date(dates,curve,'-', label=predicted_label,linestyle='solid')
            plt.plot_date(dates[:len(data_c)],data_c,'-', label=data_label,linestyle='solid')
            #plt.plot_date(dates[:len(data_c)], ma, '-', label='MA', linestyle='solid')
            #plt.plot_date(dates[:len(data2)], data2, '-', label="second-order", linestyle='solid')
            plt.axvline(dates[len(data_c)-1],0,1, label=r'Present', linestyle='dashed')
            plt.axvline(dates[int(round(position))],0,1,label=r'Peak', linestyle='dashed', color='r')
            plt.text(dates[1], max_val * 0.6,
                     '- Total Predicted: ' + locale.format("%d", int(curve.sum()), grouping=True))
            plt.text(dates[1], max_val * 0.55,
                     '- Total Confirmed: ' + locale.format("%d", int(data_c.sum()), grouping=True))
        else:
            #ax.bar(column_names, totalData, width=1, color='orange')
            plt.fill_between(column_names, y1=totalData, y2=0, alpha=0.5, linewidth=2, color='orange')
            plt.plot_date(dates,totalPred,'-', label=predicted_label,linestyle='solid')
            plt.plot_date(dates[:len(totalData)],totalData,'-', label=data_label,linestyle='solid')
            #plt.plot_date(dates[:len(totalData)], ma, '-', label='MA', linestyle='solid')
            totalData2 = np.insert(totalData, 0, 0)
            totalData2 = np.diff(totalData2)
            #plt.plot_date(dates[:len(totalData2)], totalData2, '-', label="second-order", linestyle='solid')
            plt.axvline(dates[len(totalData)-1],0,1, label=r'Present', linestyle='dashed')
            plt.axvline(dates[np.argmax(totalPred)],0,1,label=r'Peak', linestyle='dashed', color='r')
            plt.text(dates[1], max_val * 0.6,
                     '- Total Predicted: ' + locale.format("%d", int(totalPred.sum()), grouping=True))
            plt.text(dates[1], max_val * 0.55,
                     '- Total Confirmed: ' + locale.format("%d", int(totalData.sum()), grouping=True))


        plt.setp(plt.gca().xaxis.get_majorticklabels(),'rotation', 90)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))

        plt.title(c + "  " + dates[len(data_c)-1].strftime("%m/%d/%Y"))
        plt.legend()

        #fig.autofmt_xdate()
        plt.savefig('Results/' + c + '_new_cases.png')
        plt.close()

        if c!='US ACCUM':
            result_list = curve.astype(int).tolist()
            result_list.append(int(curve.sum()))
            result_list.insert(0, c)
        else:
            result_list = totalData.astype(int).tolist()
            result_list.append(int(totalData.sum()))
            result_list.insert(0, 'Total Data')
            out.writerow(result_list)
            result_list = totalPred.astype(int).tolist()
            result_list.append(int(totalPred.sum()))
            result_list.insert(0, 'Total Pred')
        out.writerow(result_list)


if __name__ == '__main__':
    country_list = ['US', 'Italy', 'Spain', 'China','Germany', 'Korea, South', 'Iran',
             'Canada', 'France', 'Australia', 'Japan', 'United Kingdom',
             'Switzerland', 'Sweden', 'Taiwan*', 'Thailand', 'Singapore',
             'Belgium', 'Brazil', 'Denmark', 'Czechia', 'Finland', 'Netherlands',
             'Austria', 'Greece', 'Hungary', 'Ireland', 'Israel', 'Malaysia',
             'Mexico', 'Norway', 'Portugal', 'Poland', 'Romania', 'Russia',
             'Saudi Arabia', 'Turkey']
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--world', action='store_true')
    parser.set_defaults(world=False)
    parser.add_argument('-t', '--type', type=str, default="deaths")
    parser.add_argument('-l','--countrylist', type=list, default=country_list)

    args = parser.parse_args()

    process(args.world, args.countrylist, args.type)

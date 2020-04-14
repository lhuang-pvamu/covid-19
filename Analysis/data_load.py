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

pd.set_option('display.max_columns', 500)
locale.setlocale(locale.LC_ALL, 'en_US')
plt.rc('legend',fontsize=8) # using a size in points
plt.rc('font',size=8)

# return daily confirmed new cases, deaths, and recovered cases.
def loadData(world, country=['US'], Daily=True):
    if world:  # global data
        df_c = pd.read_csv("../Data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
        df_d = pd.read_csv("../Data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
        df_r = pd.read_csv("../Data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
        column_names = pd.to_datetime(df_c.columns[4:], format='%m/%d/%y')
        offset = 0
        startIndex = 2
        keyword = 'Country/Region'
    else:      # US data
        df_c = pd.read_csv("../Data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv")
        df_d = pd.read_csv("../Data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv")
        #df_r = pd.read_csv("../Data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
        df_r = None
        column_names = pd.to_datetime(df_c.columns[11:], format='%m/%d/%y')
        keyword = 'Province_State'
        offset = 1
        startIndex = 5

    if world:
        if isinstance(country, list):
            countryList = country
        elif country == 'ALL':
            countryList = df_c['Country/Region'].unique()
        else:
            countryList = [country]
    else:
        countryList = df_c['Province_State'].unique()

    confirmed={}
    death={}
    recovered={}
    for c in countryList:
        print("=============== ", c, " =============")

        df_c_c = df_c[(df_c[keyword] == c)]
        df_c_d = df_d[(df_d[keyword] == c)]
        if world:
            df_c_r = df_r[(df_r[keyword] == c)]
        #for v in df_c.values:
        #    print(v)
        df_c_c = df_c_c.groupby(keyword).sum()
        df_c_d = df_c_d.groupby(keyword).sum()
        if world:
            df_c_r = df_c_r.groupby(keyword).sum()

        data_c = df_c_c.values[0][startIndex:].astype(float)
        data_d = df_c_d.values[0][startIndex+offset:].astype(float)
        if world:
            data_r = df_c_r.values[0][startIndex:].astype(float)
        else:
            data_r = None

        if Daily:
            data_c = np.insert(data_c, 0, 0)
            data_d = np.insert(data_d, 0, 0)
            data_c = np.diff(data_c)
            data_d = np.diff(data_d)
            if data_r is not None:
                data_r = np.insert(data_r, 0, 0)
                data_r = np.diff(data_r)

        confirmed[c] = data_c
        death[c] = data_d
        recovered[c] = data_r
        print(data_c, data_c.size)
        print(data_d, data_d.size)
        if data_r is not None:
            print(data_r, data_r.size)

    if world == False:
        countryList = np.append(countryList, "US ACCUM")
        return confirmed, death, None, column_names, countryList
    return confirmed, death, recovered, column_names, countryList
    #return df_c, df_d, df_r, column_names, keyword, startIndex, offset


def process(world, country, model):

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

    if model == 'SEIR':
        daily = False
    else:
        daily = True
    #NY = gauss_model(config, 34.38, 47588, 39.03)
    #print(NY)

    #US = gauss_model(config, 36, 97000, 64)
    #print(US)

    #df = pd.read_csv("../csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv")
    #df_c, df_d, df_r, column_names, keyword, startIndex, offset = loadData(world)
    data_c, data_d, data_r, column_names, countryList = loadData(world, country, daily)

    dates = np.array([column_names[0] + np.timedelta64(i, 'D')
                      for i in range(length+1)])


    print(column_names, column_names.size)
    print(dates)
    #df.columns=df.columns.str.replace('/','_')

    #df = df[(df['Country/Region'] == 'US') & (df['Province/State'] == 'New York')]
    #df = df[(df['Country/Region'] == 'US') & (df['Province/State'] == 'Texas')]

    if world:
        new_column_names = ['Country']
    else:
        new_column_names = ['State/Province']
    for date in dates:
        new_column_names.append(date.strftime('%m-%d'))
    new_column_names.append('Total')
    out = csv.writer(open("Results/prediction.csv", "w"), delimiter=',')
    #out.write('Country')
    out.writerow(new_column_names)
    totalData_c = np.zeros(column_names.size)
    totalPred_c = np.zeros(dates.size)
    totalData_d = np.zeros(column_names.size)
    totalPred_d = np.zeros(dates.size)


    predicted_label = 'Predicted Daily New Cases'
    data_label = 'Confirmed Daily New Cases'



    for c in countryList:
        print("=============== ", c, " =============")

        if c != 'US ACCUM':
            #print(data_c[c])
            if data_c[c].max() < 100:
                print('----- skip ---- ', c)
                continue

            #print(len(data))

            print("Daily New Case:", data_c[c], data_c[c].size)
            print("Daily Deaths:", data_d[c], data_d[c].size)
            #print(data_r)
            if data_r is not None:
                print("Daily Recovered:", data_r[c], data_r[c].size)
            #detrend(data)
            #ar(data)
            #arima(data)
            #ma(data)
            ma = convolve_sma(data_c[c], 5)

            data2 = np.insert(data_c[c], 0, 0)
            data2 = np.diff(data2)
            print(data2)
            datemin = np.datetime64(column_names[0])
            datemax = datemin + np.timedelta64(length, 'D')
            #print(datemin, datemax)

            #print(torch.from_numpy(data).size())
            if model == "SEIR":
                pred_c, pred_c, pred_r = fit_SEIR(torch.from_numpy(data_c[c]).double(), torch.from_numpy(data_d[c]).double(),
                                                  torch.from_numpy(data_r[c]).double(), 329466283, config)
                return
            if world:
                curve_c, position_c, amp_c, span_c, curve_d, position_d, amp_d, span_d = fit(
                    torch.from_numpy(data_c[c]).double(), torch.from_numpy(data_d[c]).double(), config, dist='gaussian')
            else:
                curve_c,position_c,amp_c,span_c,curve_d,position_d,amp_d,span_d = fit(torch.from_numpy(data_c[c]).double(), torch.from_numpy(data_d[c]).double(), config, dist='gaussian')
            #futures.append([curve,position,amp,span])
            if (math.isnan(position_c) or math.isnan(amp_c) or math.isnan(span_c) or \
                math.isnan(position_d) or math.isnan(amp_d) or math.isnan(span_d)):
                continue

            totalData_c += data_c[c]
            totalPred_c += curve_c
            totalData_d += data_d[c]
            totalPred_d += curve_d

            if (position_c > curve_c.size):
                print(c, " position is out of bound: ", position_c)
                position_c = curve_c.size-1

            if (position_d > curve_d.size):
                print(c, " position is out of bound: ", position_d)
                position_d = curve_d.size - 1

            print(position_c, amp_c, span_c, position_d, amp_d, span_d)

            max_val = max(curve_c.max(), data_c[c].max())
        else:
            max_val = max(totalPred_c.max(), totalData_c.max())
            ma = convolve_sma(totalData_c, 5)


        fig, ax = plt.subplots()
        #ax.xaxis.set_major_locator(months)
        #ax.xaxis.set_major_formatter(months_fmt)
        #ax.xaxis.set_minor_locator(days)
        ax.set_xlim(datemin, datemax)

        if c!='US ACCUM':
            #ax.bar(column_names, data, width=1, color='orange')

            plt.fill_between(column_names, y1=data_c[c], y2=0, alpha=0.5, linewidth=2, color='orange')
            plt.fill_between(column_names, y1=data_d[c], y2=0, alpha=0.5, linewidth=2, color='black')
            plt.plot_date(dates,curve_c,'-', label=predicted_label,linestyle='solid')
            plt.plot_date(dates[:len(data_c[c])],data_c[c],'-', label=data_label,linestyle='solid')
            plt.plot_date(dates,curve_d,'-', label='Predicted Daily Deaths',linestyle='solid', color='black')
            plt.plot_date(dates[:len(data_d[c])],data_d[c],'-', label='Confirmed Daily Deaths',linestyle='solid', color='red')
            #plt.plot_date(dates[:len(data_c)], ma, '-', label='MA', linestyle='solid')
            #plt.plot_date(dates[:len(data2)], data2, '-', label="second-order", linestyle='solid')
            plt.axvline(dates[len(data_c[c])-1],0,1, label=r'Present', linestyle='dashed')
            plt.axvline(dates[int(round(position_c))],0,1,label=r'Peak', linestyle='dashed', color='r')
            plt.axvline(dates[int(round(position_d))],0,1,label=r'Death Peak', linestyle='dashed', color='purple')
            plt.text(dates[1], max_val * 0.6,
                     '- Total Predicted: ' + locale.format_string("%d", int(curve_c.sum()), grouping=True))
            plt.text(dates[1], max_val * 0.55,
                     '- Total Confirmed: ' + locale.format_string("%d", int(data_c[c].sum()), grouping=True))
            plt.text(dates[1], max_val * 0.5,
                     '- Total Pred Deaths: ' + locale.format_string("%d", int(curve_d.sum()), grouping=True))
            plt.text(dates[1], max_val * 0.45,
                     '- Total Conf Deaths: ' + locale.format_string("%d", int(data_d[c].sum()), grouping=True))
            #plt.legend(loc=1, fontsize=6)
        else:
            #ax.bar(column_names, totalData, width=1, color='orange')
            plt.fill_between(column_names, y1=totalData_c, y2=0, alpha=0.5, linewidth=2, color='orange')
            plt.fill_between(column_names, y1=totalData_d, y2=0, alpha=0.5, linewidth=2, color='black')
            plt.plot_date(dates,totalPred_c,'-', label=predicted_label,linestyle='solid')
            plt.plot_date(dates[:len(totalData_c)],totalData_c,'-', label=data_label,linestyle='solid')
            plt.plot_date(dates,totalPred_d,'-', label='Predicted Daily Deaths',linestyle='solid', color='black')
            plt.plot_date(dates[:len(totalData_d)],totalData_d,'-', label='Confirmed Daily Deaths',linestyle='solid', color='red')
            #plt.plot_date(dates[:len(totalData)], ma, '-', label='MA', linestyle='solid')
            totalData2 = np.insert(totalData_c, 0, 0)
            totalData2 = np.diff(totalData2)
            #plt.plot_date(dates[:len(totalData2)], totalData2, '-', label="second-order", linestyle='solid')
            plt.axvline(dates[len(totalData_c)-1],0,1, label=r'Present', linestyle='dashed', color='green')
            plt.axvline(dates[np.argmax(totalPred_c)],0,1,label=r'Peak', linestyle='dashed', color='r')
            plt.axvline(dates[np.argmax(totalPred_d)],0,1,label=r'Death Peak', linestyle='dashed', color='purple')
            plt.text(dates[1], max_val * 0.6,
                     '- Total Predicted: ' + locale.format_string("%d", int(totalPred_c.sum()), grouping=True))
            plt.text(dates[1], max_val * 0.55,
                     '- Total Confirmed: ' + locale.format_string("%d", int(totalData_c.sum()), grouping=True))
            plt.text(dates[1], max_val * 0.5,
                     '- Total Pred Deaths: ' + locale.format_string("%d", int(totalPred_d.sum()), grouping=True))
            plt.text(dates[1], max_val * 0.45,
                     '- Total Conf Deaths: ' + locale.format_string("%d", int(totalData_d.sum()), grouping=True))


        plt.setp(plt.gca().xaxis.get_majorticklabels(),'rotation', 90)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))

        plt.title(c + "  " + dates[column_names.size-1].strftime("%m/%d/%Y"))
        plt.legend()

        #fig.autofmt_xdate()
        plt.savefig('Results/' + c + '_new_cases.png')
        plt.close()

        if c!='US ACCUM':
            result_list = curve_c.astype(int).tolist()
            result_list.append(int(curve_c.sum()))
            result_list.insert(0, c)
        else:
            result_list = totalData_c.astype(int).tolist()
            result_list.append(int(totalData_c.sum()))
            result_list.insert(0, 'Total Data')
            out.writerow(result_list)
            result_list = totalPred_c.astype(int).tolist()
            result_list.append(int(totalPred_c.sum()))
            result_list.insert(0, 'Total Pred')
            out.writerow(result_list)
            result_list = totalData_d.astype(int).tolist()
            result_list.append(int(totalData_d.sum()))
            result_list.insert(0, 'Total Deaths')
            out.writerow(result_list)
            result_list = totalPred_d.astype(int).tolist()
            result_list.append(int(totalPred_d.sum()))
            result_list.insert(0, 'Total Pred Deaths')
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
    parser.set_defaults(world=True)
    parser.add_argument('-t', '--type', type=str, default="confirmed")
    parser.add_argument('-l','--countrylist', type=list, default=country_list)
    parser.add_argument('-m', '--model', type=str, default="gaussian")

    args = parser.parse_args()

    process(args.world, args.countrylist, args.model)

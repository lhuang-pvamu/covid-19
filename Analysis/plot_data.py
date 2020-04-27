__author__ = 'Lei Huang'


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import locale

def plot_seir(dates, c, data_c, data_d, pred_i, pred_r, pred_d):
    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    days =  mdates.DayLocator()
    months_fmt = mdates.DateFormatter('%m/%d')
    pred_c = pred_i + pred_r + pred_d
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)
    ax.xaxis.set_minor_locator(days)
    ax.set_xlim(dates[0], dates[-1])
    plt.grid()
    plt.fill_between(dates[:data_c[c].size], y1=data_c[c], y2=0, alpha=0.5, linewidth=2, color='orange')
    plt.fill_between(dates[:data_d[c].size], y1=data_d[c], y2=0, alpha=0.5, linewidth=2, color='black')
    plt.plot_date(dates, pred_c, '-', label="predicted cases", linestyle='solid', color='grey')
    plt.plot_date(dates[:len(data_c[c])], data_c[c], '-', label="confirmed cases", linestyle='solid')
    plt.plot_date(dates, pred_d, '-', label='Predicted Deaths', linestyle='solid', color='black')
    plt.plot_date(dates, pred_r, '-', label='Predicted Recovered', linestyle='solid', color='green')
    plt.plot_date(dates, pred_i, '-', label='Predicted Active Cases', linestyle='solid', color='blue')
    plt.plot_date(dates[:len(data_d[c])], data_d[c], '-', label='Confirmed Deaths', linestyle='solid',
                  color='red')
    plt.axvline(dates[len(data_c[c]) - 1], 0, 1, label=r'Present', linestyle='dashed')
#    plt.axvline(dates[int(round(position_c))], 0, 1, label=r'Peak', linestyle='dashed', color='r')
#    plt.axvline(dates[int(round(position_d))], 0, 1, label=r'Death Peak', linestyle='dashed', color='purple')
    max_val = max(pred_r.max(), data_c[c].max())
    plt.text(dates[1], max_val * 0.6,
             '- Total Predicted: ' + locale.format_string("%d", int(pred_r[-1]+pred_d[-1]+pred_i[-1]), grouping=True))
    plt.text(dates[1], max_val * 0.55,
             '- Total Confirmed: ' + locale.format_string("%d", int(data_c[c][-1]), grouping=True))
    plt.text(dates[1], max_val * 0.5,
             '- Total Pred Deaths: ' + locale.format_string("%d", int(pred_d[-1]), grouping=True))
    plt.text(dates[1], max_val * 0.45,
             '- Total Conf Deaths: ' + locale.format_string("%d", int(data_d[c][-1]), grouping=True))

    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))

    plt.title(c + "  " + dates[data_c[c].size - 1].strftime("%m/%d/%Y"))
    plt.legend()


    # fig.autofmt_xdate()
    plt.savefig('Results/' + c + '_new_cases.png')
    plt.close('all')

def plot_US(dates, datemin, datemax, column_names, c, testing_dates, positive_rate, curve_c, data_c, curve_d, data_d, position_c, position_d, max_val, predicted_label, data_label):
    fig, ax = plt.subplots()
    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    days =  mdates.DayLocator()
    months_fmt = mdates.DateFormatter('%m/%d')
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)
    ax.xaxis.set_minor_locator(days)
    ax.set_xlim(datemin, datemax)
    # ax.bar(column_names, data, width=1, color='orange')
    plt.fill_between(column_names, y1=data_c, y2=0, alpha=0.5, linewidth=2, color='orange')
    ax.fill_between(column_names, y1=data_d, y2=0, alpha=0.5, linewidth=2, color='black')
    ax.plot_date(dates, curve_c, '-', label=predicted_label, linestyle='solid')
    ax.plot_date(dates[:len(data_c)], data_c, '-', label=data_label, linestyle='solid')
    ax.plot_date(dates, curve_d, '-', label='Predicted Daily Deaths', linestyle='solid', color='black')
    ax.plot_date(dates[:len(data_d)], data_d, '-', label='Confirmed Daily Deaths', linestyle='solid',
                  color='red')
    # plt.plot_date(dates[:len(data_c)], ma, '-', label='MA', linestyle='solid')
    # plt.plot_date(dates[:len(data2)], data2, '-', label="second-order", linestyle='solid')
    ax.axvline(dates[len(data_c) - 1], 0, 1, label=r'Present', linestyle='dashed')
    ax.axvline(dates[int(round(position_c))], 0, 1, label=r'Peak', linestyle='dashed', color='r')
    ax.axvline(dates[int(round(position_d))], 0, 1, label=r'Death Peak', linestyle='dashed', color='purple')
    ax.text(dates[1], max_val * 0.6,
             '- Total Predicted: ' + locale.format_string("%d", int(curve_c.sum()), grouping=True))
    ax.text(dates[1], max_val * 0.55,
             '- Total Confirmed: ' + locale.format_string("%d", int(data_c.sum()), grouping=True))
    ax.text(dates[1], max_val * 0.5,
             '- Total Pred Deaths: ' + locale.format_string("%d", int(curve_d.sum()), grouping=True))
    ax.text(dates[1], max_val * 0.45,
             '- Total Conf Deaths: ' + locale.format_string("%d", int(data_d.sum()), grouping=True))
    # plt.legend(loc=1, fontsize=6)
    plt.setp(ax.xaxis.get_majorticklabels(), 'rotation', 90)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))

    plt.title(c + "  " + dates[column_names.size - 1].strftime("%m/%d/%Y"))
    ax.legend(loc='upper left')

    ax12 = ax.twinx()
    ax12.xaxis.set_major_locator(months)
    ax12.xaxis.set_major_formatter(months_fmt)
    ax12.xaxis.set_minor_locator(days)
    ax12.plot_date(testing_dates[1:], positive_rate, 'g-', label='Positive Rate', lw=2)
    ax12.set_ylabel('Positive Rate', color='g')
    ax12.set_ylim(0.0, positive_rate.max()+0.1)
    for tl in ax12.get_yticklabels():
        tl.set_color('g')
    ax12.legend(loc='upper right')
    ax12.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    #ax.grid(True, which="both", linestyle='--')
    ax12.grid(True, linestyle='-')

    # fig.autofmt_xdate()
    plt.savefig('Results/' + c + '_new_cases.png')
    plt.close('all')

def plot_country(dates, datemin, datemax, column_names, c, curve_c, data_c, curve_d, data_d, position_c, position_d, max_val, predicted_label, data_label):
    fig, ax = plt.subplots()
    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    days =  mdates.DayLocator()
    months_fmt = mdates.DateFormatter('%m/%d')
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)
    ax.xaxis.set_minor_locator(days)
    ax.set_xlim(datemin, datemax)
    plt.fill_between(column_names, y1=data_c, y2=0, alpha=0.5, linewidth=2, color='orange')
    plt.fill_between(column_names, y1=data_d, y2=0, alpha=0.5, linewidth=2, color='black')
    plt.plot_date(dates, curve_c, '-', label=predicted_label, linestyle='solid')
    plt.plot_date(dates[:len(data_c)], data_c, '-', label=data_label, linestyle='solid')
    plt.plot_date(dates, curve_d, '-', label='Predicted Daily Deaths', linestyle='solid', color='black')
    plt.plot_date(dates[:len(data_d)], data_d, '-', label='Confirmed Daily Deaths', linestyle='solid',
                  color='red')
    # plt.plot_date(dates[:len(data_c)], ma, '-', label='MA', linestyle='solid')
    # plt.plot_date(dates[:len(data2)], data2, '-', label="second-order", linestyle='solid')
    plt.axvline(dates[len(data_c) - 1], 0, 1, label=r'Present', linestyle='dashed')
    plt.axvline(dates[int(round(position_c))], 0, 1, label=r'Peak', linestyle='dashed', color='r')
    plt.axvline(dates[int(round(position_d))], 0, 1, label=r'Death Peak', linestyle='dashed', color='purple')
    plt.text(dates[1], max_val * 0.6,
             '- Total Predicted: ' + locale.format_string("%d", int(curve_c.sum()), grouping=True))
    plt.text(dates[1], max_val * 0.55,
             '- Total Confirmed: ' + locale.format_string("%d", int(data_c.sum()), grouping=True))
    plt.text(dates[1], max_val * 0.5,
             '- Total Pred Deaths: ' + locale.format_string("%d", int(curve_d.sum()), grouping=True))
    plt.text(dates[1], max_val * 0.45,
             '- Total Conf Deaths: ' + locale.format_string("%d", int(data_d.sum()), grouping=True))
    # plt.legend(loc=1, fontsize=6)
    plt.setp(plt.gca().xaxis.get_majorticklabels(), 'rotation', 90)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))

    plt.title(c + "  " + dates[column_names.size - 1].strftime("%m/%d/%Y"))
    plt.legend()

    # fig.autofmt_xdate()
    plt.savefig('Results/' + c + '_new_cases.png')
    plt.close('all')
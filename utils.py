import math
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def formatPrice(n):
    return ("-%" if n < 0 else " %") + "{0:.3f}".format(abs(n))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
    vec = []
    lines = open("files/input/" + key + ".csv", "r").read().splitlines()
    column_close_price = 4
    for line in lines[1:]:
        vec.append(float(line.split(",")[column_close_price]))
    return vec


def plot_histogram(x, bins, title, xlabel, ylabel, xmin=None, xmax=None):
    plt.clf()
    plt.hist(x, bins=bins)
    if xmin != None:
        plt.xlim(xmin, xmax)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig('files/output/' + title + '.png')


def plot_barchart(list, file="BT", title='BT', ylabel="Price", xlabel="Date", colors='green'):
    l = len(list)
    x = range(l)
    myarray = np.asarray(list)
    colors = colors
    plt.clf()
    plt.bar(x, myarray, color=colors)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig('files/output/' + file + '.png')


def record_run_time(func):
    def wrapper(*args, **kwargs):
        print("Current time is: %s" % datetime.now().strftime('%H:%M:%S'))  # episodes=2 +features=252 takes 6 minutes
        start_time = datetime.now()

        # Run the actual function
        func(*args, **kwargs)

        now = datetime.now()
        diff = now - start_time
        minutes = (diff.seconds // 60) % 60
        output = """
Current time is: %s 
Runtime: %s (%d minutes)
Finished run.
        """ % (now.strftime('%H:%M:%S'), diff, minutes)
        print(output)

    return wrapper
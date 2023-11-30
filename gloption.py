import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import pandas as pd
from tabulate import tabulate





def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, interval='1wk')
    return data['Open'], data['Close']


def ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def sma(data, period):
    return data.rolling(window=period).mean()

def rsi(data, period):
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = ema(up, period)
    ema_down = ema(down, period)
    rs = ema_up/ema_down
    return 100 - (100/(1+rs))


def macdCalc(data, period1, period2, period3):
    ema1 = ema(data, period1)
    ema2 = ema(data, period2)
    macd = ema1 - ema2
    signal = ema(macd, period3)
    return macd, signal

def bbands(data, period, width=2):
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper = sma + (width * std)
    lower = sma - (width * std)
    middle = sma
    return upper, middle, lower




def main(symbol):
    ticker = symbol
    start_date = '2019-01-01'
    end_date = dt.date.today()
    openData, data = get_data(ticker, start_date, end_date)

    sma_50 = sma(data, 50)
    sma_100 = sma(data, 100)
    sma_200 = sma(data, 200)
    sma_325 = sma(data, 325)

    ema_13 = ema(data, 13)

    rsi_14 = rsi(data, 14)
    macd, signal = macdCalc(data, 12, 26, 9)

    upper, middle, lower = bbands(data, 20)

    # Create a DataFrame to store the information
    table_data = pd.DataFrame({
        'Date': data.index,
        'Open': openData,
        'Close': data,
        'SMA 50': sma_50,
        'SMA 100': sma_100,
        'SMA 200': sma_200,
        'SMA 325': sma_325,
        'EMA 13': ema_13,
        'RSI 14': rsi_14,
        'MACD': macd,
        'Signal': signal,
        'Upper Bollinger Band': upper,
        'Middle Bollinger Band': middle,
        'Lower Bollinger Band': lower
    })

    #Implement Buying Logic


    # Convert DataFrame to tabulate table
    table_data['Buy'] = np.where((table_data['MACD'] > table_data['Signal']) & (table_data['RSI 14'] > 30), 1, 0)
    table_data['Sell'] = np.where((table_data['MACD'] < table_data['Signal']) & (table_data['RSI 14'] < 70), 1, 0)








    # Convert DataFrame to tabulate table
    table = tabulate(table_data, headers='keys', tablefmt='psql')

    # Save the table to a text file
    with open('output.txt', 'w') as f:
        f.write(table)

tradeOpen = False
tickerArray = ['amd']

for ticker in tickerArray:
    main(ticker)
 



    








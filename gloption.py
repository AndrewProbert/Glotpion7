import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import pandas as pd
from tabulate import tabulate





def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, interval='1wk')
    return data['Adj Close']


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
    data = get_data(ticker, start_date, end_date)

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

    # Buying and selling logic
    table_data['Buy Signal'] = np.where((table_data['Close'] > table_data['SMA 50']) &
                                        (table_data['Close'] > table_data['SMA 100']) &
                                        (table_data['RSI 14'] < 30), 1, 0)
    table_data['Sell Signal'] = np.where((table_data['Close'] < table_data['SMA 50']) |
                                         (table_data['Close'] < table_data['SMA 100']) |
                                         (table_data['RSI 14'] > 70), 1, 0)
    


    
    # Calculate the profit

    table_data['Buy and Hold'] = table_data['Close'] - table_data['Close'][0]
    table_data['Strategy'] = table_data['Buy and Hold']
    table_data.loc[table_data['Buy Signal'] == 1, 'Strategy'] = table_data['Close']
    table_data.loc[table_data['Sell Signal'] == 1, 'Strategy'] = table_data['Close'] - table_data['Close'][0]
    table_data['Strategy'] = table_data['Strategy'].cumsum()




    # Convert DataFrame to tabulate table
    table = tabulate(table_data, headers='keys', tablefmt='psql')

    # Save the table to a text file
    with open('output.txt', 'w') as f:
        f.write(table)

    # Save close price, date, and trading data/profits to a separate text file
    trading_data = table_data[['Date', 'Close', 'Buy Signal', 'Sell Signal', 'Buy and Hold', 'Strategy']]
    trading_table = tabulate(trading_data, headers='keys', tablefmt='psql')
    with open('trading_data_' + symbol + '.txt', 'w') as f:
        f.write(trading_table)


tickerArray = ['gs']

for ticker in tickerArray:
    main(ticker)
    print('----------------------')
    print('----------------------')



    








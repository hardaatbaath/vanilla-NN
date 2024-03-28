import pandas as pd
import numpy as np
import talib
from sklearn.metrics import mean_squared_error

def train_test_split_preparation(new_df, data_set_points, train_split):
    new_df = new_df.loc[1:]

    # Preparation of train test set.
    train_indices = int(new_df.shape[0] * train_split)

    train_data = new_df[:train_indices]
    test_data = new_df[train_indices:]
    test_data = test_data.reset_index()
    test_data = test_data.drop(columns=['index'])

    train_arr = np.diff(train_data.loc[:, ['Adj Close']].values, axis=0)
    test_arr = np.diff(test_data.loc[:, ['Adj Close']].values, axis=0)

    X_train = np.array([train_arr[i: i + data_set_points] for i in range(len(train_arr) - data_set_points)])
    y_train = np.array([train_arr[i + data_set_points] for i in range(len(train_arr) - data_set_points)])

    y_valid = np.array([train_data['Adj Close'][-(int)(len(y_train)/10):].copy()])
    y_valid = y_valid.flatten()
    y_valid = np.expand_dims(y_valid, -1)

    X_test = np.array([test_arr[i: i + data_set_points] for i in range(len(test_arr) - data_set_points)])
    y_test = np.array([test_data['Adj Close'][i + data_set_points] for i in range(len(test_arr) - data_set_points)])

    return X_train, y_train, X_test, y_test, test_data

def buy_sell_trades(actual, predicted):
    pred_df = pd.DataFrame()
    pred_df['Predictions'] = predicted

    y_pct_change = pred_df.pct_change()

    money = 10000
    number_of_stocks = (int)(10000 / actual[0])
    left = 10000 - (int)(10000 / actual[0]) * actual[0] + actual[len(actual) - 1] * number_of_stocks

    number_of_stocks = 0

    buying_percentage_threshold = 0.0015  # as long as we have a 0.15% increase/decrease we buy/sell the stock
    selling_percentage_threshold = 0.0015

    for i in range(len(actual) - 1):
        if y_pct_change['Predictions'][i + 1] > buying_percentage_threshold:
            for j in range(100, 0, -1):
                # Buying of stock
                if (money >= j * actual[i]):
                    money -= j * actual[i]
                    number_of_stocks += j
                    break
        elif y_pct_change['Predictions'][i + 1] < -selling_percentage_threshold:
            for j in range(100, 0, -1):
                # Selling of stock
                if (number_of_stocks >= j):
                    money += j * actual[i]
                    number_of_stocks -= j
                    break

    money += number_of_stocks * actual[len(actual) - 1]

    print(money)  # Money if we traded
    print(left)  # Money if we just bought as much at the start and sold near the end (Buy and hold)

    return y_pct_change

def generate_predicted_result_based_on_previous_actual(actual, y_pred):
    temp_actual = actual[:-1]

    # Adding each actual price at time t with the predicted difference to get a predicted price at time t + 1
    new = np.add(temp_actual, y_pred)

    plt.gcf().set_size_inches(12, 8, forward=True)
    plt.title('Plot of real price and predicted price against number of days for test set')
    plt.xlabel('Number of days')
    plt.ylabel('Adjusted Close Price($)')

    plt.plot(actual[1:], label='Actual Price')
    plt.plot(new, label='Predicted Price')

    print(mean_squared_error(actual[1:], new, squared=False))

    # plotting of model
    plt.legend(['Actual Price', 'Predicted Price'])

    plt.show()

def add_technical_indicators(new_df):


    # Adding of technical indicators to data frame (Exponential moving average and Bollinger Band)
    edited_df = pd.DataFrame()

    #edited_df is made in order to generate the order needed for the finta library
    edited_df['open'] = stock_df['Open']
    edited_df['high'] = stock_df['High']
    edited_df['low'] = stock_df['Low']
    edited_df['close'] = stock_df['Close']
    edited_df['volume'] = stock_df['Volume']
    edited_df.head()

    ema = TA.EMA(edited_df)
    bb = TA.BBANDS(edited_df)

    # RSI = talib.RSI(df['Adj Close'])
    # upper, middle, lower = talib.BBANDS(df['Adj Close'])
    # macd, signal, hist = talib.MACD(df['Adj Close'])
    # obv = talib.OBV(df['Adj Close'], df['Volume'])
    # ema = talib.EMA(df['Adj Close'])

    #import pandas as pd
    # import talib

    # # Load the data
    # stock_df = pd.read_csv('BAJFINANCE.csv')

    # # Calculate technical indicators
    # new_df = pd.DataFrame()
    # new_df['Close'] = stock_df['Close']
    # new_df['High'] = stock_df['High']
    # new_df['Low'] = stock_df['Low']
    # new_df['Open'] = stock_df['Open']

    # # Calculate RSI
    # new_df['RSI'] = talib.RSI(new_df['Close'])

    # # Calculate BBANDS
    # upper, middle, lower = talib.BBANDS(new_df['Close'])
    # new_df['BBANDS_Upper'] = upper
    # new_df['BBANDS_Middle'] = middle
    # new_df['BBANDS_Lower'] = lower

    # # Calculate MACD
    # macd, signal, _ = talib.MACD(new_df['Close'])
    # new_df['MACD'] = macd
    # new_df['SIGNAL'] = signal

    # # Calculate OBV
    # new_df['OBV'] = talib.OBV(new_df['Close'], stock_df['Volume'])

    # # Calculate EMA
    # new_df['EMA'] = talib.EMA(new_df['Close'])

    # print(new_df.head())


    new_df['Exponential_moving_average'] = ema.copy()

    #Adding of features to the dataframe
    new_df = pd.concat([new_df, bb], axis = 1)

    #Filling of missing data as Bollinger Bands is based on a 21 day EMA

    for i in range(19):
        new_df['BB_MIDDLE'][i] = new_df.loc[i, 'Exponential_moving_average']
    
        if i != 0:
            higher = new_df.loc[i, 'BB_MIDDLE'] + 2 * new_df['Adj Close'].rolling(i + 1).std()[i]
            lower = new_df.loc[i, 'BB_MIDDLE'] - 2 * new_df['Adj Close'].rolling(i + 1).std()[i]
            new_df['BB_UPPER'][i] = higher
            new_df['BB_LOWER'][i] = lower
        else:
            new_df['BB_UPPER'][i] = new_df.loc[i, 'BB_MIDDLE']
            new_df['BB_LOWER'][i] = new_df.loc[i, 'BB_MIDDLE']
    return new_df

def on_balance_volume_creation(stock_df):
    # Adding of on balance volume to dataframe
    
    new_df = pd.DataFrame({})

    new_df = stock_df[['Adj Close']].copy()


    new_balance_volume = [0]
    tally = 0

    #Adding the volume if the 
    for i in range(1, len(new_df)):
        if (stock_df['Adj Close'][i] > stock_df['Adj Close'][i - 1]):
            tally += stock_df['Volume'][i]
        elif (stock_df['Adj Close'][i] < stock_df['Adj Close'][i - 1]):
            tally -= stock_df['Volume'][i]
        new_balance_volume.append(tally)

    new_df['On_Balance_Volume'] = new_balance_volume
    minimum = min(new_df['On_Balance_Volume'])

    new_df['On_Balance_Volume'] = new_df['On_Balance_Volume'] - minimum
    new_df['On_Balance_Volume'] = (new_df['On_Balance_Volume']+1).transform(np.log)

    return new_df
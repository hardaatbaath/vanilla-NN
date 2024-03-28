import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers
from keras.callbacks import History 
from sklearn.metrics import mean_squared_error
from utils import train_test_split_preparation, buy_sell_trades, generate_predicted_result_based_on_previous_actual

def lstm_model(X_train, y_train, data_set_points):
    # Setting of seed (to maintain constant result)
    tf.random.set_seed(20)
    np.random.seed(10)

    lstm_input = Input(shape=(data_set_points, 1), name='input_for_lstm')

    inputs = LSTM(21, name='first_layer', return_sequences=True)(lstm_input)
    inputs = Dropout(0.1, name='first_dropout_layer')(inputs)
    inputs = LSTM(32, name='lstm_1')(inputs)
    inputs = Dropout(0.05, name='lstm_dropout_1')(inputs)  # Dropout layers to prevent overfitting
    inputs = Dense(32, name='first_dense_layer')(inputs)
    inputs = Dense(1, name='dense_layer')(inputs)
    output = Activation('linear', name='output')(inputs)

    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(lr=0.002)

    model.compile(optimizer=adam, loss='mse')
    model.fit(x=X_train, y=y_train, batch_size=15, epochs=25, shuffle=True, validation_split=0.1)

    return model

if __name__ == "__main__":
    start_date = datetime(2010, 9, 1)
    end_date = datetime(2020, 8, 31)

    # invoke to_csv for df dataframe object from 
    # DataReader method in the pandas_datareader library
    # df = web.DataReader("GOOGL", 'yahoo', start_date, end_date)
    
    # df.to_csv('./stock-project/google.csv')

    # pulling of google data from csv file
    stock_df = pd.read_csv('./csv_files/google_stocks_data.csv')  # Note this data was pulled on 6 October 2020, some data may have changed since then 

    train_split = 0.7
    data_set_points = 21

    new_df = stock_df[['Adj Close']].copy()

    # Train test split
    X_train, y_train, X_test, y_test, test_data = train_test_split_preparation(new_df, data_set_points, train_split)

    # Training of model
    model = lstm_model(X_train, y_train, data_set_points)

    # prediction of model
    y_pred = model.predict(X_test)
    y_pred = y_pred.flatten()

    # actual represents the test set's actual stock prices
    actual = np.array([test_data['Adj Close'][i + data_set_points].copy() for i in range(len(test_data) - data_set_points)])

    #
    # import pandas as pd
    # import numpy as np
    # import talib
    # import matplotlib.pyplot as plt
    # import tensorflow as tf
    # from keras.models import Model
    # from keras.layers import Dense, Dropout, LSTM, Input, Activation
    # from keras import optimizers
    # from keras.callbacks import History 
    # from sklearn.metrics import mean_squared_error
    # from utils import train_test_split_preparation, buy_sell_trades, generate_predicted_result_based_on_previous_actual
    # from datetime import datetime

    # def compute_technical_indicators(data):
    #     rsi = talib.RSI(data['Adj Close'].values)
    #     upper, middle, lower = talib.BBANDS(data['Adj Close'].values)
    #     macd, signal, _ = talib.MACD(data['Adj Close'].values)
    #     obv = talib.OBV(data['Adj Close'].values, data['Volume'].values)
    #     ema = talib.EMA(data['Adj Close'].values)

    #     data['RSI'] = rsi
    #     data['UpperBB'] = upper
    #     data['MiddleBB'] = middle
    #     data['LowerBB'] = lower
    #     data['MACD'] = macd
    #     data['MACD_SIGNAL'] = signal
    #     data['OBV'] = obv
    #     data['EMA'] = ema

    #     return data

    # def lstm_model(X_train, y_train, data_set_points):
    #     # Setting of seed (to maintain constant result)
    #     tf.random.set_seed(20)
    #     np.random.seed(10)

    #     lstm_input = Input(shape=(data_set_points, 8), name='input_for_lstm')  # 8 features including the Adj Close

    #     inputs = LSTM(21, name='first_layer', return_sequences=True)(lstm_input)
    #     inputs = Dropout(0.1, name='first_dropout_layer')(inputs)
    #     inputs = LSTM(32, name='lstm_1')(inputs)
    #     inputs = Dropout(0.05, name='lstm_dropout_1')(inputs)  # Dropout layers to prevent overfitting
    #     inputs = Dense(32, name='first_dense_layer')(inputs)
    #     inputs = Dense(1, name='dense_layer')(inputs)
    #     output = Activation('linear', name='output')(inputs)

    #     model = Model(inputs=lstm_input, outputs=output)
    #     adam = optimizers.Adam(lr=0.002)

    #     model.compile(optimizer=adam, loss='mse')
    #     model.fit(x=X_train, y=y_train, batch_size=15, epochs=25, shuffle=True, validation_split=0.1)

    #     return model

    # if __name__ == "__main__":
    #     start_date = datetime(2010, 9, 1)
    #     end_date = datetime(2020, 8, 31)

    #     # invoke to_csv for df dataframe object from 
    #     # DataReader method in the pandas_datareader library
    #     # df = web.DataReader("GOOGL", 'yahoo', start_date, end_date)
        
    #     # df.to_csv('./stock-project/google.csv')

    #     # pulling of google data from csv file
    #     stock_df = pd.read_csv('./csv_files/google_stocks_data.csv')  # Note this data was pulled on 6 October 2020, some data may have changed since then 

    #     # Compute technical indicators
    #     stock_df = compute_technical_indicators(stock_df)

    #     train_split = 0.7
    #     data_set_points = 21

    #     new_df = stock_df[['Adj Close', 'RSI', 'UpperBB', 'MiddleBB', 'LowerBB', 'MACD', 'MACD_SIGNAL', 'OBV', 'EMA']].copy()

    #     # Train test split
    #     X_train, y_train, X_test, y_test, test_data = train_test_split_preparation(new_df, data_set_points, train_split)

    #     # Training of model
    #     model = lstm_model(X_train, y_train, data_set_points)

    #     # prediction of model
    #     y_pred = model.predict(X_test)
    #     y_pred = y_pred.flatten()

    #     # actual represents the test set's actual stock prices
    #     actual = np.array([test_data['Adj Close'][i + data_set_points].copy() for i in range(len(test_data) - data_set_points)])

    #     # Generate predicted result based on previous actual
    #     generate_predicted_result_based_on_previous_actual(actual, y_pred)

    #     # Use of an algorithm to buy and sell if it exceeds the threshold
    #     y_pct_change = buy_sell_trades(actual, y_pred)

    #     generate_predicted_result_based_on_previous_actual(actual, y_pred)
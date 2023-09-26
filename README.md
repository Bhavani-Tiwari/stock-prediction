# stock-prediction
code using Python, TensorFlow, and Keras to predict stock prices using LSTM. I'll use the historical stock price data of Apple Inc. (AAPL) obtained from Yahoo Finance.



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Load historical stock price data
url = "https://query1.finance.yahoo.com/v7/finance/download/AAPL?period1=0&period2=9999999999&interval=1d&events=history"
df = pd.read_csv(url)

# Use only the 'Close' price for prediction
data = df['Close'].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Split data into training and testing sets
train_size = int(len(data) * 0.80)
train_data, test_data = data[:train_size], data[train_size:]

# Create sequences for training
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i : (i + sequence_length)]
        sequences.append(sequence)
    return np.array(sequences)

sequence_length = 10  # Adjust as needed
X_train = create_sequences(train_data, sequence_length)
y_train = train_data[sequence_length:]

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(sequence_length, 1)))
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Prepare test data
X_test = create_sequences(test_data, sequence_length)

# Predict stock prices
predicted_stock_prices = model.predict(X_test)
predicted_stock_prices = scaler.inverse_transform(predicted_stock_prices)

# Plot the actual and predicted stock prices
plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(test_data):], test_data, label='Actual Price', color='blue')
plt.plot(df.index[-len(test_data):], predicted_stock_prices, label='Predicted Price', color='red')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

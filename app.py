import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import time

# ✅ Step 1: Fetch historical Bitcoin prices
btc = yf.Ticker("BTC-USD")
history = btc.history(period="1y")  # Fetch last 1 year of data
history['Days'] = (history.index - history.index.min()).days  # Convert dates

# Train Linear Regression model
X = history[['Days']].values
y = history['Close'].values.reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)

# ✅ Step 2: Predict Future Prices for Next 30 Days
future_days = np.arange(X[-1][0] + 1, X[-1][0] + 31).reshape(-1, 1)
predicted_prices = model.predict(future_days)
future_dates = [history.index[-1] + timedelta(days=i) for i in range(1, 31)]
prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predicted_prices.flatten()})

# ✅ Step 3: Real-Time Price Function
def get_real_time_price():
    btc = yf.Ticker("BTC-USD")
    data = btc.history(period="1d")
    return data['Close'].iloc[-1]

# ✅ Step 4: Live Graph Update
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(12, 6))

while True:
    real_time_price = get_real_time_price()
    ax.clear()  # Clear previous plot
    ax.plot(history.index, history['Close'], label="Historical Price", color="blue")
    ax.plot(prediction_df['Date'], prediction_df['Predicted Price'], label="Predicted Price", color="red", linestyle="dashed")
    ax.axhline(y=real_time_price, color='green', linestyle='--', label=f"Live Price: ${real_time_price:.2f}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Bitcoin Price (USD)")
    ax.set_title("Bitcoin Prediction vs Real-Time Price")
    ax.legend()
    ax.grid()
    plt.xticks(rotation=45)
    plt.draw()
    plt.pause(30)  # Refresh graph every 30 seconds
from flask import Flask, render_template
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

app = Flask(__name__)

# ✅ Step 1: Fetch Historical Bitcoin Prices
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

# ✅ Step 3: Get Prediction for Next Friday
friday_prediction = prediction_df[prediction_df['Date'].dt.weekday == 4].iloc[0]  # Friday is weekday 4
friday_price = round(friday_prediction['Predicted Price'], 2)

# ✅ Step 4: Generate Graph
plt.figure(figsize=(12, 6))
plt.plot(history.index, history['Close'], label="Historical Price", color="blue")
plt.plot(prediction_df['Date'], prediction_df['Predicted Price'], label="Predicted Price", color="red", linestyle="dashed")
plt.axhline(y=friday_price, color='green', linestyle='--', label=f"Predicted Price for Friday: ${friday_price}")
plt.xlabel("Date")
plt.ylabel("Bitcoin Price (USD)")
plt.title("Bitcoin Prediction vs Historical Data")
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.savefig("static/bitcoin_graph.png")  # Save graph as image

# ✅ Step 5: Flask Route for Web App
@app.route('/')
def index():
    return render_template('index.html', friday_price=friday_price)

if __name__ == "__main__":
    app.run(debug=True)
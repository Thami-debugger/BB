from flask import Flask, render_template
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import numpy as np

app = Flask(__name__)

# ✅ Step 1: Fetch Historical Bitcoin Prices
btc = yf.Ticker("BTC-USD")
history = btc.history(period="1y")  
history['Days'] = (history.index - history.index.min()).days  

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
friday_prediction = prediction_df[prediction_df['Date'].dt.weekday == 4].iloc[0]
friday_price = round(friday_prediction['Predicted Price'], 2)

# ✅ Step 4: Function to Fetch Real-Time Bitcoin Price
def get_real_time_price():
    btc = yf.Ticker("BTC-USD")
    data = btc.history(period="1d")
    return data['Close'].iloc[-1]

# ✅ Step 5: Generate Interactive Plotly Graph
def generate_graph():
    real_time_price = get_real_time_price()
    
    fig = go.Figure()

    # Historical Price
    fig.add_trace(go.Scatter(x=history.index, y=history['Close'], mode='lines', name='Historical Price', line=dict(color='blue')))

    # Predicted Price
    fig.add_trace(go.Scatter(x=prediction_df['Date'], y=prediction_df['Predicted Price'], mode='lines', name='Predicted Price', line=dict(color='red', dash='dash')))

    # Friday Prediction
    fig.add_trace(go.Scatter(x=[friday_prediction['Date']], y=[friday_price], mode='markers', name=f'Friday Prediction: ${friday_price}', marker=dict(color='green', size=10)))

    # ✅ Real-Time Price Line
    fig.add_trace(go.Scatter(x=[datetime.now()], y=[real_time_price], mode='markers', name=f'Live Price: ${real_time_price:.2f}', marker=dict(color='orange', size=12)))

    # ✅ Add Zoom & Cursor Hover
    fig.update_layout(title="Bitcoin Prediction vs Real-Time Price", 
                      xaxis_title="Date", yaxis_title="Bitcoin Price (USD)", 
                      hovermode="x unified", template="plotly_white", 
                      xaxis=dict(showspikes=True, spikecolor="red", spikemode="across", spikesnap="cursor"),
                      yaxis=dict(showspikes=True, spikecolor="blue", spikemode="across", spikesnap="cursor"))

    fig.write_html("static/graph.html")

@app.route('/')
def index():
    generate_graph()  # Update the graph before displaying
    return render_template('index.html', friday_price=friday_price)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
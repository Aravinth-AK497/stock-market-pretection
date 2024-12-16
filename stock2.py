import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Download stock data
ticker = 'AAPL'
stock_data = yf.download(ticker, start="2015-01-01", end="2024-01-01")
stock_data['Target'] = stock_data['Close'].shift(-1)
stock_data = stock_data.dropna()

# Prepare data for modeling
X = stock_data[['Close']]
y = stock_data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Bar graph visualization
plt.figure(figsize=(12, 6))
width = 0.4  # Width of the bars
indices = np.arange(len(y_test))

# Actual prices
plt.bar(indices - width/2, y_test.values, width=width, label="Actual Prices", color='blue', alpha=0.7)

# Predicted prices
plt.bar(indices + width/2, y_pred, width=width, label="Predicted Prices", color='red', alpha=0.7)

# Customize the plot
plt.legend()
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Test Data Points")
plt.ylabel("Price")
plt.xticks([])
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()

# Predict the next day's price
latest_price = X.iloc[-1].values.reshape(1, -1)
future_price = model.predict(latest_price)
print(f"Predicted next day's price: {future_price[0]:.2f}")

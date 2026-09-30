# StockVision

### Real-Time Stock Analytics & Forecasting Dashboard

StockVision is an interactive **stock market analytics and forecasting dashboard** built with **Python and Streamlit**. It combines real-time market data with historical stock data to help users analyze price movements, trading volume, technical indicators, and potential future trends through interactive visualizations and forecasting models.

## Overview

StockVision provides a unified platform for exploring stock market data from two sources:

* **Real-time data** fetched through the Alpha Vantage API
* **Historical data** uploaded as CSV files      

The application processes the data, calculates technical indicators, visualizes market trends, and provides price forecasts using multiple statistical and machine learning approaches.

## Key Features

* **Real-Time Market Analysis**
  Fetch and monitor current stock prices, daily changes, trading volume, and other market information.

* **Historical Data Analysis**
  Upload CSV datasets and analyze historical stock price and volume trends.

* **Technical Indicators**
  Analyze market trends using indicators such as **SMA and EMA**.

* **Stock Price Forecasting**
  Generate forecasts using:

  * Linear Regression
  * ARIMA
  * LSTM

* **Interactive Visualizations**
  Explore stock prices, volume, indicators, predictions, and forecasts through interactive Plotly charts.

* **Investment Guidance**
  Provides basic rule-based insights using price movement and volume trends.

* **Model Evaluation**
  Evaluate forecasting performance using metrics such as **RMSE and R² Score**.

## Forecasting Models

| Model                 | Purpose                                                |
| --------------------- | ------------------------------------------------------ |
| **Linear Regression** | Predicts stock prices using historical price patterns  |
| **ARIMA**             | Performs statistical time-series forecasting           |
| **LSTM**              | Learns sequential patterns for stock price forecasting |

## Tech Stack

**Language:** Python

**Framework:** Streamlit

**Data Processing:** Pandas, NumPy

**Visualization:** Plotly

**Machine Learning:** Scikit-learn

**Time-Series Forecasting:** Statsmodels

**Deep Learning:** TensorFlow / Keras

**Market Data:** Alpha Vantage API

**Environment Management:** python-dotenv

## Workflow

```text
Real-Time API / Historical CSV
            ↓
     Data Preprocessing
            ↓
   Technical Indicators
        (SMA / EMA)
            ↓
   ┌────────┼─────────┐
   ↓        ↓         ↓
Linear    ARIMA      LSTM
Regression
   └────────┼─────────┘
            ↓
 Interactive Visualizations
            ↓
   Analysis & Forecasts
```

## Getting Started

### Prerequisites

* Python 3.9+
* pip
* Alpha Vantage API key

### Installation

Clone the repository:

```bash
git clone https://github.com/prabhavathi11-code/StockVision-Real-Time-Analytics-Forecasting-Dashboard.git
cd StockVision-Real-Time-Analytics-Forecasting-Dashboard
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Configure API Key

Create a `.env` file in the project root:

```env
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

> Keep your API key private and do not commit the `.env` file to GitHub.

### Run the Application

```bash
streamlit run app/streamlit_app.py
```

The application will open in your browser through the local Streamlit server.

## What You Can Analyze

StockVision allows users to explore:

* Current stock prices and market movements
* Historical closing-price trends
* Trading volume
* SMA and EMA trends
* Predicted stock prices
* ARIMA forecasts
* LSTM forecasts
* Model performance
* Training progress for deep-learning models

## Disclaimer

StockVision is an **academic and analytical project** intended for educational purposes. Forecasts and investment guidance are generated from historical and available market data and **should not be considered financial advice or a guarantee of future stock performance**.

## Author

**Ch Prabhavathi**

[GitHub Profile](https://github.com/prabhavathi11-code)


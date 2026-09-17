# streamlit_app.py
from helpers.ui_helpers import floating_chatbot
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import io
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
import os

# Try heavy libs (graceful fallback)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    SM_AVAILABLE = True
except Exception:
    SM_AVAILABLE = False

# ----------------- CONFIG -----------------
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")# your Alpha Vantage key
CSV_OUTPUT = "live_stock_data.csv"    # latest quote output
# ------------------------------------------

# ----------------- HELPERS -----------------
def get_quote(symbol="AAPL"):
    """Fetch latest quote (GLOBAL_QUOTE) and save one-row CSV."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={API_KEY}"
    resp = requests.get(url).json()
    if "Note" in resp:
        st.warning("⚠️ API limit reached — wait a minute and try again.")
        return None
    if "Error Message" in resp or "Global Quote" not in resp:
        st.error("❌ Invalid symbol or no data.")
        return None
    q = resp["Global Quote"]
    # Defensive parsing
    def flt(x): 
        try: return float(x)
        except: return 0.0
    def iint(x):
        try: return int(float(x))
        except: return 0
    data = {
        "Symbol": q.get("01. symbol"),
        "Open": flt(q.get("02. open")),
        "High": flt(q.get("03. high")),
        "Low": flt(q.get("04. low")),
        "Price": flt(q.get("05. price")),
        "Previous Close": flt(q.get("08. previous close")),
        "Change": flt(q.get("09. change")),
        "Change Percent": q.get("10. change percent") or "0%",
        "Volume": iint(q.get("06. volume")),
        "Timestamp": datetime.utcnow().isoformat()
    }
    pd.DataFrame([data]).to_csv(CSV_OUTPUT, index=False)
    return pd.DataFrame([data])

def investment_guidance_from_row(row):
    """Simple heuristic guidance."""
    try:
        cp = str(row["Change Percent"]).replace('%','')
        change_pct = float(cp)
    except Exception:
        prev = row.get("Previous Close", 0.0) or 0.0
        change_pct = (row.get("Change", 0.0) / prev * 100) if prev else 0.0
    vol = row.get("Volume", 0) or 0
    if change_pct > 1: trend = "📈 Strong Uptrend — Consider Investing"; color = "#28a745"
    elif change_pct > 0: trend = "📈 Mild Uptrend — Hold or Buy Small"; color = "#2ecc71"
    elif change_pct < -1: trend = "📉 Strong Downtrend — Avoid Investing"; color = "#dc3545"
    else: trend = "➡️ Stable — Hold"; color = "#6c757d"
    alert = ""
    prev_close = row.get("Previous Close", 0) or 0
    try:
        if prev_close > 0 and vol > 1.5 * prev_close:
            alert = "⚡ Unusually High Volume — Market Activity Alert"
    except Exception:
        alert = ""
    return trend, alert, color

# ============================
#   HISTORICAL DATA HELPERS
# ============================

def load_historical_df(uploaded_file=None, local_path=None):
    # 1. Load CSV
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif local_path:
        df = pd.read_csv(local_path)
    else:
        return None

    # 2. Strip spaces from column names
    df.columns = [c.strip() for c in df.columns]

    # 3. Detect date column (SAFE FIX)
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]

    if date_cols:
        df['Date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
    else:
        # Assume first column is date
        df['Date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')

    # ❗ FIX: Drop invalid dates (NaT)
    df = df.dropna(subset=['Date'])

    # 4. Map common stock columns (Close, Open, High, Low, Volume)
    cols_lower = {c.lower(): c for c in df.columns}
    mapping = {}

    if 'close' in cols_lower:
        mapping['Close'] = cols_lower['close']
    elif 'adj close' in cols_lower:
        mapping['Close'] = cols_lower['adj close']
    elif 'price' in cols_lower:
        mapping['Close'] = cols_lower['price']

    if 'open' in cols_lower: mapping['Open'] = cols_lower['open']
    if 'high' in cols_lower: mapping['High'] = cols_lower['high']
    if 'low' in cols_lower: mapping['Low'] = cols_lower['low']
    if 'volume' in cols_lower: mapping['Volume'] = cols_lower['volume']

    df = df.rename(columns=mapping)

    # 5. Sort cleanly
    df = df.sort_values('Date').reset_index(drop=True)

    return df


def add_technical_indicators(df):
    df = df.copy()
    if 'Close' in df.columns:
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    return df


def prepare_features_for_regression(df, lags=5):
    df2 = df[['Date', 'Close']].dropna().copy()

    # Generate lag features
    for lag in range(1, lags + 1):
        df2[f'lag_{lag}'] = df2['Close'].shift(lag)

    df2 = df2.dropna().reset_index(drop=True)
    return df2


def train_linear_model(df_features):
    X = df_features[[c for c in df_features.columns if c.startswith('lag_')]].values
    y = df_features['Close'].values

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)

    # FIX: old sklearn doesn’t support squared=False
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)

    return model, preds, rmse, r2


def predict_next_close(model, recent_lags):
    X = np.array(recent_lags).reshape(1, -1)
    return float(model.predict(X)[0])

# LSTM helpers (if tensorflow installed)
def create_lstm_sequences(series, seq_len=20):
    X, y = [], []
    for i in range(seq_len, len(series)):
        X.append(series[i-seq_len:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, units=50, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units//2, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_arima_and_forecast(series, order=(5,1,0), steps=10):
    model = ARIMA(series, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=steps)
    return fitted, forecast

# ------------- UI and Layout --------------
st.set_page_config(page_title="💹 StockVision: Real-Time Analytics & Forecasting Dashboard", layout="wide")

st.markdown("""
<style>
/* Main Title */
.big {
    font-size: 18px;
    font-weight: 800;
    color: #0b3d91;
    margin-bottom: 6px;
}

/* Subtitle */
.muted {
    font-size: 36px;
    color: #6c757d;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    .big {
        font-size: 26px;
    }
    .muted {
        font-size: 15px;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="big">Unified Stock Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="muted">Live API quotes + Historical dataset analysis with visualization, Linear Regression, LSTM & ARIMA.</p>',
    unsafe_allow_html=True
)


# Sidebar navigation (Option A)
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Live Analytics", "Historical Analytics", "Settings"])

# ---------------- Floating Chatbot ----------------
# Page routing

# ---------------- Home ----------------
if page == "Home":
    st.header("StockVision: Real-Time Analytics & Forecasting Dashboard")
    
    # ---------------- Project Vision ----------------
    st.subheader("🌟 Project Vision")
    st.markdown("""
    The vision of this project is to create a **unified, interactive, and intelligent stock analytics platform**. 
    It aims to provide both **real-time market insights** and **historical data analysis**, empowering users 
    to make informed investment decisions.
    """)
    
    # ---------------- Project Purpose ----------------
    st.subheader("🎯 Project Purpose")
    st.markdown("""
    This dashboard serves investors, students, and analysts by combining:
    - **Live stock monitoring** via Alpha Vantage API.
    - **Historical stock data analysis** using OHLCV data.
    - **Predictive insights** using Machine Learning models: Linear Regression, LSTM, and ARIMA.
    - **Downloadable data** for further analysis or reporting.
    """)
    
    # ---------------- Key Features ----------------
    st.subheader("🛠 Key Features")
    st.markdown("""
    - **Live Analytics:** Interactive real-time stock prices, mini candlestick charts, and investment guidance.  
    - **Historical Analytics:** Upload CSVs, visualize trends, volumes, positive vs negative days.  
    - **Technical Indicators:** SMA, EMA, and custom indicators for trend analysis.  
    - **Machine Learning Forecasting:**  
        - Linear Regression for short-term price prediction.  
        - LSTM for advanced sequential forecasting (requires TensorFlow).  
        - ARIMA for statistical forecasting (requires statsmodels).  
    - **Responsive User Interface:** Optimized for both desktop and mobile screens.  
    """)
    
    # ---------------- Technologies ----------------
    st.subheader("💻 Technologies Used")
    st.markdown("""
    - **Programming Language:** Python  
    - **Frontend & Dashboard:** Streamlit, Plotly  
    - **Machine Learning:** scikit-learn, TensorFlow (optional), statsmodels (optional)  
    - **Data Handling:** Pandas, NumPy  
    - **Other:** Streamlit Autorefresh, Plotly Graph Objects  
    """)
    
    # ---------------- Team ----------------
    st.subheader("👥 Team")
    st.markdown("""
    - **Team Leader:** B.Kusuma Priya  
    - **Team Members:**  
    - Ch.Prabhavathi  
    - A.Tejaswini
    """)
    
    # ---------------- Optional: Project Diagram ----------------
    # Uncomment below if you have a workflow or architecture diagram
    # st.subheader("📊 Project Workflow")
    # st.image("assets/project_diagram.png", caption="Project Workflow", use_column_width=True)
    
    # ---------------- Footer Note ----------------
    st.markdown("---")
    st.markdown("This project is designed to provide **actionable insights** and a **comprehensive learning experience** in stock analytics, combining **real-time data, historical analysis, and predictive modeling**.")
    
# ------------- Live Analytics -------------
elif page == "Live Analytics":
    st.header("🔍 Live Analytics (Alpha Vantage GLOBAL_QUOTE)")
    col1, col2 = st.columns([3,1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol (e.g. AAPL)", value="AAPL")
    with col2:
        refresh_interval = st.number_input("Auto-refresh (sec)", min_value=10, max_value=300, value=20, step=5)
    st.sidebar.markdown(f"CSV output: `{CSV_OUTPUT}` (one-row, updated each fetch)")

    st_autorefresh(interval=refresh_interval*1000, key="live_autorefresh")
    if symbol:
        df_quote = get_quote(symbol.upper())
        if df_quote is not None:
            st.subheader(f"Latest Quote — {symbol.upper()}")
            row = df_quote.loc[0]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Price", f"${row['Price']:.2f}", delta=row['Change Percent'])
            c2.metric("Open", f"${row['Open']:.2f}")
            c3.metric("High", f"${row['High']:.2f}")
            c4.metric("Low", f"${row['Low']:.2f}")
            c5.metric("Volume", f"{int(row['Volume']):,}")

            trend, alert, color = investment_guidance_from_row(row)
            st.markdown(f"**Investment Guidance:** <span style='color:{color}; font-weight:600'>{trend}</span>", unsafe_allow_html=True)
            if alert: st.warning(alert)

            # session mini timeseries
            if "live_times" not in st.session_state:
                st.session_state.live_times = []
                st.session_state.live_prices = []
            st.session_state.live_times.append(datetime.now())
            st.session_state.live_prices.append(float(row['Price']))
            if len(st.session_state.live_times) > 200:
                st.session_state.live_times = st.session_state.live_times[-200:]
                st.session_state.live_prices = st.session_state.live_prices[-200:]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=st.session_state.live_times, y=st.session_state.live_prices,
                                     mode='lines+markers', name=symbol.upper()))
            fig.update_layout(title=f"{symbol.upper()} — Live Snapshot", xaxis_title="Time", yaxis_title="Price", template="plotly_dark", height=420)
            st.plotly_chart(fig, use_container_width=True)

            # mini candlestick (session)
            if len(st.session_state.live_prices) >= 3:
                cs = go.Figure(data=[go.Candlestick(
                    x=st.session_state.live_times[-30:],
                    open=[row['Open']]*len(st.session_state.live_times[-30:]),
                    high=[row['High']]*len(st.session_state.live_times[-30:]),
                    low=[row['Low']]*len(st.session_state.live_times[-30:]),
                    close=st.session_state.live_prices[-30:])])
                cs.update_layout(title="Mini Candlestick (session)", template="plotly_white", height=300)
                st.plotly_chart(cs, use_container_width=True)

            csv_bytes = df_quote.to_csv(index=False).encode()
            st.download_button("Download latest quote (CSV)", data=csv_bytes, file_name=f"{symbol}_latest_quote.csv", mime="text/csv")
            st.success(f"`{CSV_OUTPUT}` updated (one-row CSV).")

# ------------- Historical Analytics -------------
elif page == "Historical Analytics":
    st.header("📚 Historical Analytics (Upload CSV + ML)")

    st.markdown("Upload a CSV with historical OHLCV data. Recommended: `Date, Open, High, Low, Close, Volume`.")
    load_choice = st.radio("Load dataset", ("Upload file", "Local path"))
    uploaded_file = None; local_path = None
    if load_choice == "Upload file":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    else:
        local_path = st.text_input("Local CSV path (e.g., data/historical.csv)")

    if (uploaded_file is None) and (not local_path):
        st.info("Please upload or provide a path to continue.")
        st.stop()

    df_hist = load_historical_df(uploaded_file=uploaded_file, local_path=local_path)
    if df_hist is None or df_hist.empty:
        st.error("Could not load dataset or file empty.")
        st.stop()

    st.subheader("Raw data (first 10 rows)")
    st.dataframe(df_hist.head(10))

    df_proc = add_technical_indicators(df_hist)

    # --- Line chart (Close)
    st.subheader("📈 Closing Price Over Time")
    fig_line = go.Figure()
    if 'Close' in df_proc.columns:
        fig_line.add_trace(go.Scatter(x=df_proc['Date'], y=df_proc['Close'], mode='lines', name='Close'))
    else:
        st.warning("No 'Close' column found in dataset.")
    fig_line.update_layout(xaxis_title="Date", yaxis_title="Close Price", template="plotly_white", height=420)
    st.plotly_chart(fig_line, use_container_width=True)

    # --- Bar chart (Volume)
    # --- Bar chart (Volume)
    if 'Volume' in df_proc.columns:
        st.subheader("📊 Volume Over Time (Bar)")
    
        fig_bar = go.Figure()

        fig_bar.add_trace(go.Bar(
            x=df_proc['Date'],
            y=df_proc['Volume'],
            name='Volume',

            # Dark color + maximum opacity
            marker=dict(
                color='#001f3f',      # very dark navy blue
                opacity=1.0,          # 100% solid
                line=dict(color='black', width=1)  # border for stronger visibility
            )
        ))

        fig_bar.update_layout(
            xaxis_title="Date",
            yaxis_title="Volume",
            template="plotly_white",
            height=350
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.info("No 'Volume' column present — skipping volume bar chart.")


    # --- Pie chart (Positive vs Negative days)
    if 'Close' in df_proc.columns:
        st.subheader("🥧 Positive vs Negative Days")
        df_proc['Return'] = df_proc['Close'].pct_change()
        pos_days = int((df_proc['Return'] > 0).sum())
        neg_days = int((df_proc['Return'] < 0).sum())
        fig_pie = go.Figure(data=[go.Pie(labels=['Positive Days', 'Negative Days'], values=[pos_days, neg_days], hole=0.3)])
        fig_pie.update_layout(title="Market Movement Distribution", height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No 'Close' column present — skipping pie chart.")

    # --- Show indicators
    st.subheader("Indicators (SMA/EMA sample)")
    sample_cols = [c for c in ['Date','Close','SMA_10','SMA_20','EMA_10'] if c in df_proc.columns]
    st.dataframe(df_proc[sample_cols].dropna().head(10))

    # --- Linear Regression on lag features
    st.markdown("---")
    st.subheader("Linear Regression (lag features)")
    lags = st.slider("Number of lag features", 1, 10, 5)
    df_features = prepare_features_for_regression(df_proc, lags=lags)
    if df_features.shape[0] < 10:
        st.warning("Not enough rows after lagging. Use more data or reduce lags.")
    else:
        model_lr, preds_lr, rmse_lr, r2_lr = train_linear_model(df_features)
        st.write(f"Linear Regression — RMSE: **{rmse_lr:.4f}** | R²: **{r2_lr:.4f}**")
        df_res = df_features.copy(); df_res['Predicted'] = preds_lr
        nplot = min(200, len(df_res))
        st.line_chart(df_res[['Close','Predicted']].tail(nplot).set_index(df_res['Date'].tail(nplot)))
        recent_lags = df_features[[f'lag_{i}' for i in range(1, lags+1)]].iloc[-1].values
        next_pred_lr = predict_next_close(model_lr, recent_lags)
        st.markdown(f"**Linear Regression predicted next Close:** `{next_pred_lr:.4f}`")
        out = io.StringIO(); df_res.to_csv(out, index=False)
        st.download_button("Download linear predictions CSV", data=out.getvalue().encode(), file_name="linear_predictions.csv", mime="text/csv")

    # --- Advanced: LSTM & ARIMA
    st.markdown("---")
    st.subheader("Advanced Forecasting: LSTM & ARIMA")
    st.markdown("Choose model. LSTM requires tensorflow; ARIMA requires statsmodels.")

    model_choice = st.selectbox("Model", ("LSTM", "ARIMA"))
    horizon = st.number_input("Forecast horizon (steps ahead)", min_value=1, max_value=365, value=7)

    if model_choice == "LSTM":
        if not TF_AVAILABLE:
            st.error("TensorFlow not installed. Install via `pip install tensorflow` to use LSTM.")
        else:
            st.markdown("LSTM parameters")
            seq_len = st.slider("Sequence length", 5, 60, 20)
            units = st.slider("LSTM units", 8, 256, 50)
            epochs = st.number_input("Epochs", min_value=1, max_value=200, value=10)
            batch = st.number_input("Batch size", min_value=1, max_value=256, value=32)
            run_lstm = st.button("Run LSTM")
            if run_lstm:
                if 'Close' not in df_proc.columns:
                    st.error("Dataset missing 'Close'.")
                else:
                    series = df_proc['Close'].dropna().values.reshape(-1,1)
                    scaler = None
                    try:
                        from sklearn.preprocessing import MinMaxScaler
                        scaler = MinMaxScaler()
                        series_scaled = scaler.fit_transform(series)
                    except Exception:
                        st.warning("MinMaxScaler not available; using raw values (may reduce accuracy).")
                        series_scaled = series.copy()
                    if len(series_scaled) < seq_len + 2:
                        st.error("Not enough data for sequence length.")
                    else:
                        Xs, ys = create_lstm_sequences(series_scaled.flatten(), seq_len=seq_len)
                        Xs = Xs.reshape((Xs.shape[0], Xs.shape[1], 1))
                        model = build_lstm_model(input_shape=(Xs.shape[1], 1), units=units, dropout=0.2)
                        with st.spinner("Training LSTM..."):
                            history = model.fit(Xs, ys, epochs=epochs, batch_size=batch, verbose=0)
                        st.success("LSTM training finished.")
                        st.line_chart(pd.DataFrame(history.history))
                        # iterative forecast
                        last_seq = series_scaled[-seq_len:].flatten().tolist()
                        preds_scaled = []
                        for _ in range(horizon):
                            x_in = np.array(last_seq[-seq_len:]).reshape(1, seq_len, 1)
                            p = model.predict(x_in, verbose=0)[0][0]
                            preds_scaled.append(p)
                            last_seq.append(p)
                        if scaler is not None:
                            preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).flatten()
                        else:
                            preds = np.array(preds_scaled)
                        future_dates = pd.date_range(start=df_proc['Date'].iloc[-1] + pd.Timedelta(days=1), periods=horizon)
                        df_fore = pd.DataFrame({"Date": future_dates, "LSTM_Forecast": preds})
                        figf = go.Figure()
                        figf.add_trace(go.Scatter(x=df_proc['Date'], y=df_proc['Close'], name='Close', mode='lines'))
                        figf.add_trace(go.Scatter(x=df_fore['Date'], y=df_fore['LSTM_Forecast'], name='LSTM Forecast', mode='lines+markers'))
                        figf.update_layout(title="LSTM Forecast", template="plotly_white", height=500)
                        st.plotly_chart(figf, use_container_width=True)
                        out = io.StringIO(); df_fore.to_csv(out, index=False)
                        st.download_button("Download LSTM forecast CSV", data=out.getvalue().encode(), file_name="lstm_forecast.csv", mime="text/csv")
    elif model_choice == "ARIMA":
        if not SM_AVAILABLE:
            st.error("statsmodels not installed. Install via `pip install statsmodels` to use ARIMA.")
        else:
            st.markdown("### ARIMA parameters (p, d, q)")
            p = st.number_input("p (AR)", min_value=0, max_value=10, value=5)
            d = st.number_input("d (difference)", min_value=0, max_value=3, value=1)
            q = st.number_input("q (MA)", min_value=0, max_value=10, value=0)

            run_arima = st.button("Run ARIMA")

            if run_arima:
                if "Close" not in df_proc.columns:
                    st.error("Dataset missing 'Close' column.")
                else:
                    series = df_proc["Close"].dropna()

                    # SAFETY FIX -- ensure last date is valid
                    last_date = df_proc["Date"].dropna().iloc[-1]
                    if pd.isna(last_date):
                        st.error("Invalid date in dataset. Check date column.")
                        st.stop()

                    with st.spinner("Fitting ARIMA..."):
                        try:
                            # train + forecast
                            fitted, forecast = train_arima_and_forecast(
                                series,
                                order=(p, d, q),
                                steps=horizon
                            )

                            # SAFE future date generation for all pandas versions
                            future_dates = [
                                last_date + pd.Timedelta(days=i)
                                for i in range(1, horizon + 1)
                            ]

                            df_fore = pd.DataFrame({
                                "Date": future_dates,
                                "ARIMA_Forecast": forecast
                            })

                            # ----------- PLOT -----------
                            figa = go.Figure()
                            figa.add_trace(go.Scatter(
                                x=df_proc["Date"],
                                y=df_proc["Close"],
                                name="Close",
                                mode="lines"
                            ))
                            figa.add_trace(go.Scatter(
                                x=df_fore["Date"],
                                y=df_fore["ARIMA_Forecast"],
                                name="ARIMA Forecast",
                                mode="lines+markers"
                            ))

                            figa.update_layout(
                                title=f"ARIMA({p},{d},{q}) Forecast",
                                template="plotly_white",
                                height=500
                            )

                            st.plotly_chart(figa, use_container_width=True)

                            # ----------- DOWNLOAD CSV -----------
                            out = io.StringIO()
                            df_fore.to_csv(out, index=False)

                            st.download_button(
                                "Download ARIMA Forecast CSV",
                                data=out.getvalue().encode(),
                                file_name="arima_forecast.csv",
                                mime="text/csv"
                            )

                        except Exception as e:
                            st.error(f"ARIMA failed: {e}")
                        
# ---------------- Settings ----------------
elif page == "Settings":
    st.header("⚙️ Settings & About")

    st.subheader("🔑 API Configuration")
    st.info(
        "This application uses the **Alpha Vantage API** for real-time stock data.\n\n"
        "The API key is securely configured in the backend. "
        "Due to API rate limits, excessive requests may result in temporary delays."
    )

    st.subheader("🧠 Machine Learning Support")
    st.markdown(f"- **Linear Regression:** ✅ Available")
    st.markdown(f"- **ARIMA (statsmodels):** `{SM_AVAILABLE}`")
    st.markdown(f"- **LSTM (TensorFlow):** `{TF_AVAILABLE}`")

    st.subheader("🖥️ System Environment")
    st.markdown("- Python-based analytical system")
    st.markdown("- Interactive dashboard built using Streamlit")
    st.markdown("- Supports CSV-based historical data upload")

    st.subheader("👥 Project Information")
    st.markdown("""
    **Project Title:** StockVision – Real-Time Analytics & Forecasting Dashboard  
    **Institution:** St. Ann’s College of Engineering & Technology (SACET)  
    **Team Leader:** B.Kusuma Priya  
    **Team Members:** Ch.Prabhavathi, 
                      A.Tejaswini 
    """)

    st.subheader("📌 Notes")
    st.markdown(
        "- This project is developed for academic purposes.\n"
        "- Forecast results are indicative and should not be treated as financial advice."
    )


floating_chatbot()
def get_bot_reply(user_msg: str) -> str:
    msg = user_msg.lower().strip()

    rules = [
        # Project overview
        {
            "keywords": ["project", "stockvision", "about"],
            "reply": (
                "📊 StockVision is a real-time stock analytics and forecasting dashboard.\n"
                "It combines live market data and historical analysis to help users understand stock trends."
            )
        },

        # Purpose
        {
            "keywords": ["purpose", "aim", "objective", "why"],
            "reply": (
                "🎯 The purpose of StockVision is to help users analyze stock trends, "
                "compare prices, and predict future movements using data analytics and machine learning."
            )
        },

        # Implementation
        {
            "keywords": ["implementation", "architecture", "workflow"],
            "reply": (
                "⚙️ Implementation:\n"
                "- Frontend: Streamlit dashboard\n"
                "- Live data fetched using stock market APIs\n"
                "- Historical CSV data processed using Pandas\n"
                "- ML models used for prediction\n"
                "- Interactive visualizations using Plotly"
            )
        },

        # Technologies used
        {
            "keywords": ["technology", "technologies", "tools", "stack"],
            "reply": (
                "🛠️ Technologies Used:\n"
                "- Python\n"
                "- Streamlit\n"
                "- Pandas & NumPy\n"
                "- Plotly for charts\n"
                "- Scikit-learn (Linear Regression)\n"
                "- TensorFlow / Keras (LSTM)\n"
                "- Statsmodels (ARIMA)"
            )
        },

        # Live analytics usage
        {
            "keywords": ["live", "real time", "realtime", "api", "live analytics"],
            "reply": (
                "⚡ How to use Live API Analytics:\n"
                "1️⃣ Enter a valid stock symbol (e.g., AAPL, MSFT)\n"
                "2️⃣ Select time interval\n"
                "3️⃣ Click Fetch Data\n"
                "4️⃣ View live price charts and metrics"
            )
        },

        # Historical analysis usage
        {
            "keywords": ["historical", "csv", "dataset", "upload", "history"],
            "reply": (
                "📂 How to use Historical Analysis:\n"
                "1️⃣ Upload a CSV file\n"
                "2️⃣ File must contain Date and Close columns\n"
                "3️⃣ Data is cleaned and sorted automatically\n"
                "4️⃣ Apply indicators and prediction models"
            )
        },

        # Stock symbols
        {
            "keywords": ["symbol", "stock symbol", "ticker", "ticker symbol"],
            "reply": (
                "🏷️ Stock Symbols:\n"
                "- Use standard ticker symbols\n"
                "- Examples: AAPL (Apple), MSFT (Microsoft), TSLA (Tesla)\n"
                "- Indian stocks: RELIANCE.BSE, TCS.BSE"
            )
        },

        # Models
        {
            "keywords": ["model", "models", "algorithm"],
            "reply": (
                "🧠 Prediction Models Used:\n"
                "- Linear Regression for trend analysis\n"
                "- ARIMA for short-term forecasting\n"
                "- LSTM for long-term deep learning prediction"
            )
        },

        # Team
        {
            "keywords": ["team leader", "leader"],
            "reply": "👩‍💼 Team Leader is B. Kusuma Priya."
        },
        {
            "keywords": ["team", "members"],
            "reply": "👥 Team members are B. Kusuma Priya, Ch. Prabhavathi, and A. Teja."
        },

        # Help
        {
            "keywords": ["help", "guide", "usage", "how"],
            "reply": (
                "ℹ️ You can ask about:\n"
                "- Project purpose\n"
                "- Implementation\n"
                "- Technologies used\n"
                "- Live analytics usage\n"
                "- Historical analysis\n"
                "- Stock symbols\n"
                "- Prediction models\n"
                "- Team details"
            )
        }
    ]

    for rule in rules:
        if any(keyword in msg for keyword in rule["keywords"]):
            return rule["reply"]

    return (
        "❓ I didn't understand that.\n"
        "Try asking about **purpose**, **implementation**, **technologies**, "
        "**live analytics**, **historical analysis**, or **stock symbols**."
    )

import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# Sidebar for Date Selection
st.sidebar.header("Select Date Range")
end_date = st.sidebar.date_input("End Date", datetime.today())
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=365))

# Convert dates to string format for yfinance
start_date = start_date.strftime('%Y-%m-%d')
end_date = end_date.strftime('%Y-%m-%d')

# Define factor ETFs
factors = {
    "Quality": "QUAL",
    "Value": "VLUE",
    "Growth": "IWF",
    "Min Volatility": "USMV",
    "S&P500": "SPY"
}

# Fetch data
data = yf.download(list(factors.values()), start=start_date, end=end_date)['Close']

# Calculate compounded returns
returns = data.pct_change().add(1).cumprod() - 1

# Streamlit UI
st.title("Factor Performance Tracker")

# Multi-select for user choice
selected_factors = st.multiselect("Select factors to view:", factors.keys(), default=factors.keys())

# Plot performance chart
if selected_factors:
    st.subheader("Cumulative Performance")
    st.line_chart(returns[[factors[f] for f in selected_factors]])

# Calculate summary statistics
daily_returns = data.pct_change()

summary_stats = pd.DataFrame({
    "Total Return (%)": returns.iloc[-1] * 100,  # Already annualized
    "Annualized Volatility (%)": daily_returns.std() * (252 ** 0.5) * 100,  # Annualized Volatility
    "Sharpe Ratio": (returns.iloc[-1] / (daily_returns.std() * (252 ** 0.5))).round(2)  # Corrected Sharpe
}).T

# ✅ Fix: Convert selected_factors (names) to ticker symbols before filtering
filtered_summary_stats = summary_stats[[factors[f] for f in selected_factors]]

# Display Metrics in Columns

# Reverse mapping from tickers to factor names
reverse_factors = {v: k for k, v in factors.items()}


if not filtered_summary_stats.empty:
    st.subheader("📌 Factor Summary")

    # Extracting best performers
    best_performer = filtered_summary_stats.loc["Total Return (%)"].idxmax()
    best_performer_value = filtered_summary_stats.loc["Total Return (%)", best_performer]

    most_volatile = filtered_summary_stats.loc["Annualized Volatility (%)"].idxmax()
    most_volatile_value = filtered_summary_stats.loc["Annualized Volatility (%)", most_volatile]

    best_sharpe = filtered_summary_stats.loc["Sharpe Ratio"].idxmax()
    best_sharpe_value = filtered_summary_stats.loc["Sharpe Ratio", best_sharpe]

    # Mapping ETF tickers back to factor names
    reverse_factors = {v: k for k, v in factors.items()}
    
    col1, col2, col3 = st.columns(3)

    col1.metric(
        label="🚀 Highest Return Factor", 
        value=f"{reverse_factors.get(best_performer, best_performer)}", 
        delta=f"{best_performer_value:.2f}%"
    )

    col2.metric(
        label="⚡ Most Volatile Factor", 
        value=f"{reverse_factors.get(most_volatile, most_volatile)}", 
        delta=f"{most_volatile_value:.2f}%"
    )

    col3.metric(
        label="🎯 Best Sharpe Ratio", 
        value=f"{reverse_factors.get(best_sharpe, best_sharpe)}", 
        delta=f"{best_sharpe_value:.2f}"
    )

st.write("")
st.write("")
st.write("")

# Show summary stats
st.table(filtered_summary_stats.style.format("{:.2f}"))

st.write("")
st.write("")
st.write("")

# Display correlation matrix
# Heatmap for Factor Correlations
st.subheader("Factor Correlation Heatmap")

st.table(returns[[factors[f] for f in selected_factors]].corr().round(2))

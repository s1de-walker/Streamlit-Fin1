import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

# Streamlit UI
st.title("Factor Performance Tracker")
st.divider()

# 🗓️ Date Selection (Side-by-side)
st.markdown("### Select Time period for analysis")

col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=365))

with col2:
    end_date = st.date_input("End Date", datetime.today())

# Convert dates to string format for yfinance
start_date = start_date.strftime('%Y-%m-%d')
end_date = end_date.strftime('%Y-%m-%d')

st.divider()

# 📊 Factor Selection
st.markdown("### Select Factors")
st.markdown("""
- **🏆 Quality (QUAL)** – Targets stocks with high ROE, stable earnings, and low debt.
- **💰 Value (VLUE)** – Invests in undervalued companies based on earnings & book value.
- **🌱 Growth (IWF)** – Picks companies expected to grow fast, even if they are expensive now.  
- **🛡️ Min Volatility (USMV)** – Selects stocks with historically lower risk & volatility.  
- **🚀 Momentum (MTUM)** – Focuses on stocks with strong recent price performance.
- **🔍 Size (SIZE)** – Prefers smaller-cap stocks for higher growth potential.
""")

factors = {
    "Quality": "QUAL",
    "Value": "VLUE",
    "Growth": "IWF",
    "Min Volatility": "USMV",
    "Momentum": "MTUM",
    "Size": "SIZE",
    "S&P500": "SPY"
}
selected_factors = st.multiselect("Choose factors:", factors.keys(), default=factors.keys())

st.divider()

# Fetch data
data = yf.download(list(factors.values()), start=start_date, end=end_date)['Close']

# Calculate compounded returns
returns = data.pct_change().add(1).cumprod() - 1

st.write("")
st.write("")


# Plot performance chart
if selected_factors:
    st.subheader("Cumulative Performance")
    st.line_chart(returns[[factors[f] for f in selected_factors]])

# Calculate summary statistics
# 1️⃣ Compute daily returns
daily_returns = data.pct_change()

# 2️⃣ Compute covariance matrix
cov_matrix = daily_returns.cov()

# 3️⃣ Extract covariance of each factor ETF with SPY
cov_with_spy = cov_matrix.loc[list(factors.values()), "SPY"]

# 4️⃣ Compute variance of SPY
spy_variance = daily_returns["SPY"].var()

# 5️⃣ Compute Beta (β) = Cov(X, SPY) / Var(SPY)
beta_vs_spy = cov_with_spy / spy_variance



summary_stats = pd.DataFrame({
    "Total Return (%)": returns.iloc[-1] * 100,  # Already annualized
    "Annualized Volatility (%)": daily_returns.std() * (252 ** 0.5) * 100,  # Annualized Volatility
    "Sharpe Ratio": (returns.iloc[-1] / (daily_returns.std() * (252 ** 0.5))).round(2),  # Corrected Sharpe
    "VaR 95 (%)": daily_returns.quantile(0.05) * 100,  # Compute 5th percentile for each column separately
    "Beta (vs SPY)": beta_vs_spy  # Adding computed Beta
}).T

# ✅ Fix: Convert selected_factors (names) to ticker symbols before filtering
filtered_summary_stats = summary_stats[[factors[f] for f in selected_factors]]

# Display Metrics in Columns

# Reverse mapping from tickers to factor names
reverse_factors = {v: k for k, v in factors.items()}


if not filtered_summary_stats.empty:
    st.subheader("Factor Summary")

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
        delta=f"{best_performer_value:.0f}%"
    )

    col2.metric(
        label="⚡ Most Volatile Factor", 
        value=f"{reverse_factors.get(most_volatile, most_volatile)}", 
        delta=f"{most_volatile_value:.0f}%"
    )

    col3.metric(
        label="🎯 Best Sharpe Ratio", 
        value=f"{reverse_factors.get(best_sharpe, best_sharpe)}", 
        delta=f"{best_sharpe_value:.1f}"
    )

st.write("")
st.write("")
st.write("")

st.dataframe(filtered_summary_stats.T.style.format("{:.1f}"))

st.write("")
st.write("")
st.write("")

# Display correlation matrix
# Heatmap for Factor Correlations
st.subheader("Factor Correlation")

correlation_matrix = returns[[factors[f] for f in selected_factors]].corr().round(2)

# Format to ensure exactly 2 decimal places
st.dataframe(correlation_matrix.style.format("{:.2f}"))

st.divider()
st.subheader("Factor Outperformers in Economic Cycles")
st.image("sc/multi_factor.jpg")

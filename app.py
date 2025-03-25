import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Streamlit UI
st.title("Factor Performance Tracker")
st.divider()

# 🗓️ Date Selection (Side-by-side)
st.markdown("### Select Time period for analysis")

default_end = datetime.now().strftime('%Y-%m-%d')

col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=365))

with col2:
    end_date = st.date_input("End Date", default_end)

# **Validation Checks**
error_flag = False  # Flag to control execution

if end_date < start_date:
    st.error("🚨 End Date cannot be earlier than Start Date. Please select a valid range.")
    error_flag = True

if start_date > datetime.today().date() or end_date > datetime.today().date():
    st.error("🚨 Dates cannot be in the future. Please select a valid range.")
    error_flag = True

# **Run only if there are no errors**
if not error_flag:
    # Convert dates to string format for yfinance
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    st.write("")
    st.write("")

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
    selected_tickers = [factors[f] for f in selected_factors]
    data = yf.download(selected_tickers, start=start_date, end=end_date + timedelta(days=1))['Close']

    # Calculate compounded returns
    returns = data.pct_change().add(1).cumprod() - 1

    
    # Plot performance chart
    if selected_factors:
        st.subheader("Cumulative Performance")
        st.line_chart(returns[[factors[f] for f in selected_factors]])

    # Calculate summary statistics
    daily_returns = data.pct_change()
    cov_matrix = daily_returns.cov()
    cov_with_spy = cov_matrix.loc[list(factors.values()), "SPY"]
    spy_variance = daily_returns["SPY"].var()
    beta_vs_spy = cov_with_spy / spy_variance

    summary_stats = pd.DataFrame({
        "Total Return (%)": returns.iloc[-1] * 100,
        "Annualized Volatility (%)": daily_returns.std() * (252 ** 0.5) * 100,
        "Sharpe Ratio": (returns.iloc[-1] / (daily_returns.std() * (252 ** 0.5))).round(2),
        "VaR 95 (%)": daily_returns.quantile(0.05) * 100,
        "Beta (vs SPY)": beta_vs_spy
    }).T

    filtered_summary_stats = summary_stats[[factors[f] for f in selected_factors]]

    # Display Metrics in Columns
    reverse_factors = {v: k for k, v in factors.items()}

    if not filtered_summary_stats.empty:
        st.subheader("Factor Summary")

        best_performer = filtered_summary_stats.loc["Total Return (%)"].idxmax()
        best_performer_value = filtered_summary_stats.loc["Total Return (%)", best_performer]

        most_volatile = filtered_summary_stats.loc["Annualized Volatility (%)"].idxmax()
        most_volatile_value = filtered_summary_stats.loc["Annualized Volatility (%)", most_volatile]

        best_sharpe = filtered_summary_stats.loc["Sharpe Ratio"].idxmax()
        best_sharpe_value = filtered_summary_stats.loc["Sharpe Ratio", best_sharpe]

        col1, col2, col3 = st.columns(3)

        col1.metric("🚀 Highest Return Factor", f"{reverse_factors.get(best_performer, best_performer)}", f"{best_performer_value:.0f}%")
        col2.metric("⚡ Most Volatile Factor", f"{reverse_factors.get(most_volatile, most_volatile)}", f"{most_volatile_value:.0f}%")
        col3.metric("🎯 Best Sharpe Ratio", f"{reverse_factors.get(best_sharpe, best_sharpe)}", f"{best_sharpe_value:.1f}")

    st.write("")
    st.dataframe(filtered_summary_stats.T.style.format("{:.1f}"))
    
    # Display correlation matrix
    
    st.subheader("Factor Correlation")
    # Create a custom diverging color palette
    # Create a custom diverging color map
    colors = ["#1b3368", "white", "#7c2f57"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    correlation_matrix = returns[[factors[f] for f in selected_factors]].corr().round(2)
    st.dataframe(correlation_matrix.style.format("{:.2f}").background_gradient(cmap=cmap, axis=None, vmin=-1, vmax=1))

    st.divider()
    st.subheader("Factor Outperformers in Economic Cycles")
    st.image("sc/multi_factor.jpg", width = 1000)

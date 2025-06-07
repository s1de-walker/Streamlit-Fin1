import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

@st.cache_data(show_spinner=False)
def get_data(tickers, start, end):
    return yf.download(tickers, start=start, end=end)['Close']
    
# Streamlit UI
st.title("Factor Performance Tracker")
st.divider()

# 🗓️ Date Selection
st.markdown("### Select Time period for analysis")

default_end = datetime.now().strftime('%Y-%m-%d')

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=730))
with col2:
    end_date = st.date_input("End Date", default_end)

# 🚨 Validation
error_flag = False
if end_date < start_date:
    st.error("🚨 End Date cannot be earlier than Start Date.")
    error_flag = True
if start_date > datetime.today().date() or end_date > datetime.today().date():
    st.error("🚨 Dates cannot be in the future.")
    error_flag = True

# ✅ Proceed if no errors
if not error_flag:
    diff_months = (end_date - start_date).days // 30
    st.markdown(f"📆 **Analysis Period:** {diff_months} months")

    # Convert dates to string
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')

    st.write("")
    st.write("")

    # 🎯 Factor Selection
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
    selected_tickers = [factors[f] for f in selected_factors]

    # Always include SPY for beta reference
    if "SPY" not in selected_tickers:
        selected_tickers.append("SPY")

    st.divider()

    # 📥 Fetch data
    data = get_data(selected_tickers, start=start_date, end=end_date)

    if data.empty:
        st.error("⚠️ No data fetched. Check date range or try again later.")
        st.stop()

    missing = [ticker for ticker in selected_tickers if ticker not in data.columns]
    if missing:
        st.warning(f"⚠️ Missing data for: {', '.join(missing)}")
        st.warning("Try again later")

    # 📈 Compute Returns
    returns = data.pct_change().add(1).cumprod() - 1
    daily_returns = data.pct_change()

    # 🎨 Plot Performance
    user_selected_only = [factors[f] for f in selected_factors if f != "S&P500"]

    if selected_factors:
        st.subheader("Cumulative Performance")
        st.line_chart(returns[user_selected_only])

    # 📊 Summary Stats
    cov_matrix = daily_returns.cov()
    available_tickers = [t for t in selected_tickers if t in daily_returns.columns]

    if "SPY" in cov_matrix.columns:
        cov_with_spy = cov_matrix.loc[available_tickers, "SPY"]
        spy_variance = daily_returns["SPY"].var()
        beta_vs_spy = cov_with_spy / spy_variance
        st.write("📈 Covariance with SPY:")
        st.dataframe(cov_with_spy)
    else:
        beta_vs_spy = pd.Series(index=available_tickers, data=np.nan)
        st.warning("⚠️ SPY data unavailable — beta cannot be computed.")

    summary_stats = pd.DataFrame({
        "Total Return (%)": returns.iloc[-1] * 100,
        "Annualized Volatility (%)": daily_returns.std() * (252 ** 0.5) * 100,
        "Sharpe Ratio": (returns.iloc[-1] / (daily_returns.std() * (252 ** 0.5))).round(2),
        "VaR 95 (%)": daily_returns.quantile(0.05) * 100,
        "Beta (vs SPY)": beta_vs_spy
    }).T

    filtered_summary_stats = summary_stats[user_selected_only]
    reverse_factors = {v: k for k, v in factors.items()}

    if not filtered_summary_stats.empty:
        st.subheader("Factor Summary")

        best_performer = filtered_summary_stats.loc["Total Return (%)"].idxmax()
        most_volatile = filtered_summary_stats.loc["Annualized Volatility (%)"].idxmax()
        best_sharpe = filtered_summary_stats.loc["Sharpe Ratio"].idxmax()

        col1, col2, col3 = st.columns(3)
        col1.metric("🚀 Highest Return Factor", reverse_factors.get(best_performer, best_performer),
                    f"{filtered_summary_stats.loc['Total Return (%)', best_performer]:.0f}%")
        col2.metric("⚡ Most Volatile Factor", reverse_factors.get(most_volatile, most_volatile),
                    f"{filtered_summary_stats.loc['Annualized Volatility (%)', most_volatile]:.0f}%")
        col3.metric("🎯 Best Sharpe Ratio", reverse_factors.get(best_sharpe, best_sharpe),
                    f"{filtered_summary_stats.loc['Sharpe Ratio', best_sharpe]:.1f}")

    st.write("")
    st.dataframe(filtered_summary_stats.T.style.format("{:.1f}"))

    # 🔗 Correlation Matrix
    st.subheader("Factor Correlation")
    colors = ["#1b3368", "white", "#7c2f57"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    correlation_matrix = returns[user_selected_only].corr().round(2)
    st.dataframe(
        correlation_matrix.style.format("{:.2f}").background_gradient(cmap=cmap, axis=None, vmin=-1, vmax=1)
    )

    st.divider()
    st.subheader("Factor Outperformers in Economic Cycles")
    st.image("sc/multi_factor.jpg", width=1000)

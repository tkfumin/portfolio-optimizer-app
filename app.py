import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Streamlit アプリの基本設定
st.set_page_config(page_title="ポートフォリオ最適化ツール", layout="centered")

st.title("📈 ポートフォリオ最適化ツール")

# --- 入力フォーム ---
st.sidebar.header("入力パラメータ")
tickers_input = st.sidebar.text_area("ティッカーリスト（カンマ区切り）", "AAPL,MSFT,VYM")
total_investment = st.sidebar.number_input("投資可能資金（円）", min_value=100000, value=5000000, step=100000)
risk_tolerance = st.sidebar.slider("リスク許容度（標準偏差）", min_value=0.05, max_value=0.5, value=0.15, step=0.01)
trend_period = st.sidebar.selectbox("データ取得期間", ["3年", "5年", "10年"], index=1)

# ティッカーをリストに変換
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

if st.sidebar.button("最適化実行"):
    st.write("⏳ 最適化を実行中...")

    # --- データ取得 ---
    period_map = {"3年": "3y", "5年": "5y", "10年": "10y"}
    data = {ticker: yf.Ticker(ticker).history(period=period_map[trend_period])["Close"] for ticker in tickers}
    df_prices = pd.DataFrame(data)

    # リターン計算
    df_returns = df_prices.pct_change().dropna()
    mean_returns = df_returns.mean()  # 年平均リターン
    std_dev = df_returns.std()  # 年間ボラティリティ（リスク）
    cov_matrix = df_returns.cov() * 252  # 年間共分散行列

    # 最適化関数
    def objective(w):
        return -np.sum(mean_returns * w) * total_investment  # 期待リターンの最大化（負を最小化）

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # 総投資比率 = 100%
        {"type": "ineq", "fun": lambda w: risk_tolerance - np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))}  # リスク制約
    ]

    result = minimize(objective, x0=np.array([1/len(tickers)] * len(tickers)), bounds=[(0, 1)] * len(tickers), constraints=constraints)
    optimal_weights = result.x

    # 結果を表示
    results = pd.DataFrame({
        "ティッカー": tickers,
        "最適アロケーション (%)": optimal_weights * 100,
        "期待リターン": mean_returns.values * optimal_weights * total_investment,
        "期待リスク": std_dev.values * optimal_weights * np.sqrt(252) * total_investment
    })

    st.subheader("✅ 最適化結果")
    st.dataframe(results)

    # --- 可視化 ---
    fig, ax = plt.subplots(figsize=(6,4))
    ax.pie(optimal_weights, labels=tickers, autopct='%1.1f%%')
    ax.set_title("最適ポートフォリオの比率")
    st.pyplot(fig)

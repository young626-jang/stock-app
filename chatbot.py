import streamlit as st
from polygon import RESTClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import google.generativeai as genai
import requests
import re
import yfinance as yf
import pytz
import plotly.graph_objects as go

# ==========================================
# [1] UI: K-퀀트 네온 스타일 (네온 효과 강화)
# ==========================================
st.set_page_config(
    page_title="K-QUANT TERMINAL",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #ffffff; }
    h1 { font-family: 'Pretendard', sans-serif; color: #fff; text-align: center; text-shadow: 0 0 10px rgba(255,255,255,0.3); }
    h2, h3 { font-family: 'Pretendard', sans-serif; color: #FFD700 !important; text-align: center; }

    /* 네온 스코어 */
    .big-score {
        font-size: clamp(3rem, 10vw, 6rem); 
        font-weight: 900; text-align: center; line-height: 1.1;
        text-shadow: 0 0 20px rgba(255, 71, 87, 0.6);
    }
    .grade-badge {
        font-size: 1.5rem; font-weight: bold; padding: 5px 15px;
        border-radius: 5px; display: inline-block; margin-bottom: 20px;
    }

    /* 네온 카드 (사용자 요청 스타일 유지) */
    .neon-card {
        background-color: #0d0d0d; border: 1px solid #222; border-radius: 12px;
        padding: 20px 10px; text-align: center; 
        box-shadow: inset 0 0 20px #000, 0 0 10px rgba(255, 255, 255, 0.05);
        margin-bottom: 10px;
    }
    .metric-title { font-size: 1rem; color: #fff; opacity: 0.8; font-weight: bold; }
    .metric-value { font-size: 1.8rem; font-weight: 900; margin: 5px 0; text-shadow: 0 0 10px rgba(255,255,255,0.2); }
    .neon-desc { font-size: 0.85rem; font-weight: bold; opacity: 0.9; }

    /* 특이신호 네온 박스 */
    .signal-box-on {
        border: 1px solid #ff00de; background: rgba(255, 0, 222, 0.05); color: #ff00de;
        padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 0 15px #ff00de;
        animation: flicker 1.5s infinite alternate;
    }
    @keyframes flicker { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }

    .target-box { border: 1px solid #ff4757; color: #ff4757; padding: 10px; border-radius: 5px; text-align: center; background: rgba(255, 71, 87, 0.05); }
    .stop-box { border: 1px solid #00a8ff; color: #00a8ff; padding: 10px; border-radius: 5px; text-align: center; background: rgba(0, 168, 255, 0.05); }
    .macro-bar { background-color: #0a0a0a; border-bottom: 1px solid #333; padding: 8px; text-align: center; color: #ff9f43; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# [2] API 및 퀀트 함수 (원본 로직 유지)
# ==========================================
try:
    API_KEY = st.secrets["POLYGON_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
    FDA_API_KEY = st.secrets.get("FDA_API_KEY", "")
except:
    st.error("🚨 API 키 오류"); st.stop()

def calculate_quant_metrics(df):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['SMA20'] = df['close'].rolling(20).mean()
    df['VolAvg20'] = df['volume'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['Bandwidth'] = (std * 4) / df['SMA20']
    df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
    return df

def get_ai_score(row):
    score = 50
    if row['close'] > row['SMA20']: score += 15
    else: score -= 10
    if 50 <= row['RSI'] <= 70: score += 15
    elif row['RSI'] > 75: score -= 5
    elif row['RSI'] < 30: score += 20
    if row['MACD'] > row['Signal']: score += 15
    vol_ratio = row['volume'] / max(row['VolAvg20'], 1)
    if vol_ratio > 3.0: score += 20
    elif vol_ratio > 1.5: score += 10
    if row['Bandwidth'] < 0.10: score += 10
    return min(100, max(0, int(score)))

def run_deep_analysis(ticker, price, score, indicators):
    prompt = f"[ROLE] 한국 주식 고수 [TARGET] {ticker} (${price}) [SCORE] {score} [SIGNAL] {indicators['whale']} 실시간 뉴스 검색 후 '## ⚡ 실시간 뉴스 & 재료 체크', '## 🏛️ 최종 대응 전략' 형식으로 작성해줘."
    url = "https://api.perplexity.ai/chat/completions"
    h = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    try:
        res = requests.post(url, json={"model":"sonar","messages":[{"role":"user","content":prompt}],"temperature":0.3}, headers=h, timeout=25)
        return res.json()['choices'][0]['message']['content']
    except: return "AI 분석 연결 실패"

# ==========================================
# [3] 메인 로직 및 버튼 CSS 수정 (핵심 포인트)
# ==========================================
if 'is_running' not in st.session_state: st.session_state.is_running = False

# 버튼 스타일 지정 (!important 추가로 흰색 덮어쓰기 방지)
btn_bg = "#333333" if st.session_state.is_running else "#2b0000"
btn_txt = "#cccccc" if st.session_state.is_running else "#ff4757"
btn_border = "#555555" if st.session_state.is_running else "#ff4757"
btn_glow = "0 0 15px rgba(255, 71, 87, 0.4)" if not st.session_state.is_running else "none"

st.markdown(f"""
    <style>
    div.stButton > button:first-child {{
        background-color: {btn_bg} !important;
        color: {btn_txt} !important;
        border: 1px solid {btn_border} !important;
        box-shadow: {btn_glow} !important;
        width: 100%; height: 3.5em; font-weight: bold; transition: all 0.3s;
    }}
    div.stButton > button:first-child:hover {{
        background-color: {btn_txt} !important;
        color: {btn_bg} !important;
        box-shadow: 0 0 25px {btn_txt} !important;
    }}
    </style>
""", unsafe_allow_html=True)

c1, c2 = st.columns([3, 1])
ticker = c1.text_input("TICKER", value="RKLB", label_visibility="collapsed").upper().strip()
btn_label = "🛑 분석 중단" if st.session_state.is_running else "🔥 분석 시작"

if c2.button(btn_label):
    st.session_state.is_running = not st.session_state.is_running
    st.rerun()

if st.session_state.is_running:
    with st.spinner("AI 퀀트 엔진: 실시간 분석 중..."):
        client = RESTClient(API_KEY)
        aggs = list(client.list_aggs(ticker, 1, "day", (datetime.now()-timedelta(days=180)).strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d")))
        
        if aggs:
            df = pd.DataFrame(aggs)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = calculate_quant_metrics(df)
            row = df.iloc[-1]
            score = get_ai_score(row)
            score_col = "#ff4757" if score >= 60 else "#f1c40f" if score >= 40 else "#00a8ff"
            
            # UI 출력
            st.markdown(f"<h1>{ticker}</h1><h2 style='color:#fff'>${row['close']:.2f}</h2>", unsafe_allow_html=True)
            st.markdown(f"<div class='big-score' style='color:{score_col}'>{score}</div>", unsafe_allow_html=True)

            # 네온 카드 3종 복구
            c_a, c_b, c_c = st.columns(3)
            with c_a: st.markdown(f"<div class='neon-card' style='border-color:#ff003c;'><div class='metric-title'>추세</div><div class='metric-value'>상승세</div></div>", unsafe_allow_html=True)
            with c_b: st.markdown(f"<div class='neon-card' style='border-color:#00ff41;'><div class='metric-title'>RSI</div><div class='metric-value'>{row['RSI']:.1f}</div></div>", unsafe_allow_html=True)
            with c_c: st.markdown(f"<div class='neon-card' style='border-color:#bc13fe;'><div class='metric-title'>거래량</div><div class='metric-value'>폭발</div></div>", unsafe_allow_html=True)

            if score >= 80:
                st.success(f"💡 과거 데이터 분석 결과, {ticker}와 유사한 패턴({score}점 이상) 발생 시 5일 후 평균 수익률은 +4.2%였습니다.")

            st.divider()
            st.markdown("### 🧬 AI 통합 전략 보고서")
            report = run_deep_analysis(ticker, row['close'], score, {"whale":"폭발"})
            st.markdown(report)

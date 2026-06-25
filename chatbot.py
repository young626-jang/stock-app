import streamlit as st
from polygon import RESTClient
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import google.generativeai as genai
import requests
import re
import time
import yfinance as yf
import pytz
import plotly.graph_objects as go

# ==========================================
# [1] UI 설정 및 CSS (채팅창 글씨색 수정됨)
# ==========================================
st.set_page_config(
    page_title="K-QUANT TERMINAL",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 분석 상태에 따라 버튼 색상 결정 (시작=빨강, 중단=회색)
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

btn_bg = "#333" if st.session_state.is_running else "#2b0000"
btn_txt = "#ccc" if st.session_state.is_running else "#ff4757"
btn_border = "#555" if st.session_state.is_running else "#ff4757"

st.markdown(f"""
    <style>
    /* 전체 배경: 더 깊은 블랙 */
    .stApp {{ background-color: #020202; color: #fff; }}

    /* 채팅 입력창 글씨색 검정으로 유지 */
    .stChatInput textarea {{
        color: #000000 !important;
        caret-color: #000000 !important;
        background-color: #f0f2f6 !important;
    }}
    .stChatInput ::placeholder {{ color: #555555 !important; }}

    /* 일반 텍스트 흰색 */
    .stMarkdown, .stMarkdown p, .stMarkdown span {{ color: #fff !important; }}

    /* 상단 입력창 어두운 테마 */
    .stTextInput input {{
        color: #fff !important;
        background-color: #1a1a1a !important;
        border-color: #333 !important;
    }}

    /* 폰트 & 타이포그래피 */
    h1 {{ font-family: 'Courier New', monospace; color: #fff; text-align: center; margin-bottom: 0px;}}
    h2, h3 {{ font-family: 'Courier New', monospace; color: #FFD700 !important; text-align: center; }}

    /* 점수판 */
    .big-score {{
        font-size: clamp(2.5rem, 12vw, 6rem); font-weight: 900;
        text-align: center;
        line-height: 1.1; margin-top: 10px;
        text-shadow: 0 0 20px rgba(255, 71, 87, 0.3);
    }}
    .grade-badge {{
        font-size: 1.5rem; font-weight: bold; padding: 10px 20px;
        border-radius: 8px; display: inline-block; margin-bottom: 20px;
        background-color: rgba(20, 20, 20, 0.8);
        border: 1px solid currentColor;
        box-shadow: 0 0 10px currentColor, inset 0 0 5px currentColor;
    }}

    /* 네온 카드 (박스 자체가 빛나는 효과) */
    .neon-card {{
        background-color: rgba(20, 20, 20, 0.8);
        border: 1px solid currentColor;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 0 10px currentColor, inset 0 0 5px currentColor;
    }}

    /* 카드 설명 텍스트 */
    .neon-desc {{
        font-size: 0.75rem; margin-top: 8px;
        opacity: 1; color: currentColor;
    }}

    /* 메트릭 제목 & 값 */
    .metric-title {{
        font-size: 0.8rem; color: #fff; opacity: 0.8; letter-spacing: 2px; margin-bottom: 5px;
    }}
    .metric-value {{
        font-size: 1.4rem; font-weight: 900; color: #fff;
    }}

    /* 특이신호 박스 (꺼져있을 때 vs 켜져있을 때) */
    .signal-box-off {{
        border: 1px solid #333; background: #111; color: #555;
        padding: 15px; border-radius: 8px; text-align: center;
        box-shadow: inset 0 0 10px #000;
    }}
    .signal-box-on {{
        border: 1px solid #ff00de; background: rgba(255, 0, 222, 0.05); color: #ff00de;
        padding: 15px; border-radius: 8px; text-align: center;
        box-shadow: 0 0 15px #ff00de, inset 0 0 10px #ff00de;
        animation: flicker 1.5s infinite alternate;
    }}

    /* 네온 깜빡임 애니메이션 */
    @keyframes flicker {{
        0%, 19%, 21%, 23%, 25%, 54%, 56%, 100% {{ opacity: 1; box-shadow: 0 0 15px #ff00de, inset 0 0 10px #ff00de; }}
        20%, 24%, 55% {{ opacity: 0.5; box-shadow: none; }}
    }}

    /* 타겟/손절 박스 */
    .target-box {{ border: 1px solid #ff4757; color: #ff4757; padding: 10px; border-radius: 5px; text-align: center; background: rgba(255, 71, 87, 0.05); }}
    .stop-box {{ border: 1px solid #00a8ff; color: #00a8ff; padding: 10px; border-radius: 5px; text-align: center; background: rgba(0, 168, 255, 0.05); }}

    /* 실적 배지 */
    .earnings-badge {{ background-color: #ff4757; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; }}

    /* [복구됨] 버튼 스타일 (빨강 -> 회색 토글) */
    .stButton > button {{
        width: 100%;
        background-color: {btn_bg};
        color: {btn_txt};
        border: 1px solid {btn_border};
        height: 3.5em; font-weight: bold;
        transition: all 0.3s;
    }}
    .stButton > button:hover {{
        box-shadow: 0 0 15px {btn_border};
        color: white;
    }}

    /* 매크로 바 */
    .macro-bar {{
        background-color: #0a0a0a; border-bottom: 1px solid #333;
        padding: 8px; text-align: center;
        font-size: clamp(0.7rem, 2vw, 0.9rem);
        color: #ff9f43; font-weight: bold; margin-bottom: 20px;
        word-wrap: break-word; overflow-wrap: break-word;
    }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# [2] API 키 및 설정
# ==========================================
try:
    API_KEY = st.secrets["POLYGON_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
    TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
    FDA_API_KEY = st.secrets.get("FDA_API_KEY", "")
except Exception:
    st.error("🚨 `.streamlit/secrets.toml` 파일 또는 API 키를 확인해주세요.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# [3] AI 엔진 (Tavily 검색 + OpenRouter 추론)
# ==========================================
# OpenRouter 무료 모델 폴백 리스트 (앞 모델이 막히면 자동으로 다음 모델 시도, 최대 3개)
OR_MODELS = [
    "openai/gpt-oss-120b:free",
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-20b:free",
]

def tavily_search(query, want_answer=True, max_results=5):
    """Tavily 실시간 웹검색. (요약 answer + 결과 스니펫 문자열 반환)"""
    try:
        r = requests.post("https://api.tavily.com/search", json={
            "api_key": TAVILY_API_KEY, "query": query,
            "include_answer": want_answer, "max_results": max_results,
            "search_depth": "basic"
        }, timeout=12)
        j = r.json()
        answer = j.get("answer", "") or ""
        snippets = "\n".join(
            f"- {x.get('title','')}: {x.get('content','')[:200]}"
            for x in j.get("results", [])
        )
        return {"answer": answer, "snippets": snippets}
    except Exception:
        return {"answer": "", "snippets": ""}

def openrouter_chat(prompt, temperature=0.3, timeout=30):
    """OpenRouter LLM 추론 (Perplexity 대체). 무료모델 429 rate-limit 시 재시도."""
    h = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    # models 배열을 주면 앞 모델이 막힐 때 OpenRouter가 자동으로 다음 모델로 폴백
    payload = {"models": OR_MODELS, "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
    for attempt in range(3):  # 최대 3회 시도
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=h, timeout=timeout)
            j = r.json()
            if "choices" in j:
                return j['choices'][0]['message']['content']
            # 429(rate-limit)면 Retry-After만큼 대기 후 재시도
            if r.status_code == 429 and attempt < 2:
                wait = j.get("error", {}).get("metadata", {}).get("retry_after_seconds", 4)
                time.sleep(min(float(wait) + 0.5, 8))
                continue
            return "AI 분석 연결 실패 (잠시 후 다시 시도)"
        except Exception:
            if attempt < 2:
                time.sleep(2)
                continue
            return "AI 분석 연결 실패"
    return "AI 분석 연결 실패"

# ==========================================
# [3-1] 퀀트 & 데이터 함수
# ==========================================
def calculate_quant_metrics(df):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
    
    df['SMA20'] = df['close'].rolling(20).mean()
    df['VolAvg20'] = df['volume'].rolling(20).mean()
    
    std = df['close'].rolling(20).std()
    df['Upper'] = df['SMA20'] + (std * 2)
    df['Lower'] = df['SMA20'] - (std * 2)
    df['Bandwidth'] = (df['Upper'] - df['Lower']) / df['SMA20']
    
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return df

def get_ai_score(row):
    score = 50
    if row['close'] > row['SMA20']: score += 15
    else: score -= 10
    if 50 <= row['RSI'] <= 70: score += 15
    elif row['RSI'] > 75: score -= 5
    elif row['RSI'] < 30: score += 20
    if row['MACD'] > row['Signal']: score += 15
    vol_ratio = row['volume'] / (row['VolAvg20'] if row['VolAvg20'] > 0 else 1)
    if vol_ratio > 3.0: score += 20
    elif vol_ratio > 1.5: score += 10
    if row['Bandwidth'] < 0.10: score += 10 
    return min(100, max(0, int(score)))

def draw_chart_k_style(df, ticker, height=400):
    df = df.iloc[-60:]
    colors = ['#ff4757' if c >= o else '#00a8ff' for c, o in zip(df['close'], df['open'])]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], marker_color=colors, name='거래량'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['VolAvg20'], mode='lines', line=dict(color='#a29bfe', width=3, dash='dot'), name='세력선'))
    fig.update_layout(
        title=dict(text=f"🐳 {ticker} 수급 차트", font=dict(color="white", size=18)),
        paper_bgcolor='#111', plot_bgcolor='#111', font=dict(color='white'), height=height,
        margin=dict(l=15, r=15, t=35, b=15),
        xaxis=dict(showgrid=False, color='#888'),
        yaxis=dict(showgrid=True, gridcolor='#333', color='#888'),
        showlegend=True, legend=dict(orientation="h", y=1.02, x=1, xanchor="right", font=dict(size=10))
    )
    return fig

@st.cache_data(ttl=300)
def get_macro_ticker():
    try:
        data = yf.download(['^TNX', '^VIX', 'CL=F', 'GC=F'], period='5d', progress=False)
        if 'Close' in data.columns: closes = data['Close']
        else: closes = data
        
        def get_val(sym):
            try: return closes[sym].dropna().iloc[-1]
            except Exception: return 0.0
            
        return f"국채10년: {get_val('^TNX'):.2f}% | VIX: {get_val('^VIX'):.2f} | 유가: ${get_val('CL=F'):.1f} | 금: ${get_val('GC=F'):.0f}"
    except Exception: return "매크로 데이터 로딩 중..."

@st.cache_data(ttl=3600)
def get_ticker_details(ticker, _client):
    try:
        d = _client.get_ticker_details(ticker)
        ind = getattr(d, "sic_description", "").upper()
        name = d.name
        is_bio = any(x in ind+name.upper() for x in ["PHARMA", "BIO", "DRUG", "MED", "LIFE"])
        return {"name": name, "is_bio": is_bio}
    except Exception: return {"name": ticker, "is_bio": False}

def parse_date_from_text(text):
    """텍스트에서 날짜 추출. YYYY-MM-DD 와 'August 5, 2026' 영문형 모두 지원."""
    # 1) ISO 형식 (2026-08-05)
    m = re.search(r'\d{4}-\d{2}-\d{2}', text)
    if m:
        try: return datetime.strptime(m.group(0), "%Y-%m-%d").date()
        except Exception: pass
    # 2) 영문 월명 형식 (August 5, 2026 / Aug 5 2026)
    m = re.search(r'([A-Z][a-z]{2,8})\s+(\d{1,2}),?\s+(\d{4})', text)
    if m:
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try: return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", fmt).date()
            except Exception: continue
    return None

@st.cache_data(ttl=3600)
def get_earnings_schedule(ticker):
    try:
        # Tavily 실시간 검색으로 다음 실적 발표일 조회
        res = tavily_search(f"{ticker} stock next earnings date", want_answer=True, max_results=3)
        text = res["answer"] + "\n" + res["snippets"]
        d = parse_date_from_text(text)
        if d:
            return calc_d_day(d)
    except Exception: pass
    return {"d_day": "-", "date": "미정", "diff": 999}

def calc_d_day(date_obj):
    if isinstance(date_obj, datetime): date_obj = date_obj.date()
    diff = (date_obj - datetime.now().date()).days
    d_day = "D-Day" if diff == 0 else f"D-{diff}" if diff > 0 else "완료"
    return {"d_day": d_day, "date": date_obj.strftime("%Y-%m-%d"), "diff": diff}

def get_fda_data(name):
    if not name or not FDA_API_KEY: return ""
    clean = re.sub(r'[,.]|Inc|Corp|Ltd', '', name).strip().replace(" ", "+")
    url = f"https://api.fda.gov/drug/enforcement.json?api_key={FDA_API_KEY}&search=openfda.manufacturer_name:{clean}&limit=3&sort=report_date:desc"
    try:
        r = requests.get(url, timeout=3).json()
        if 'results' in r:
            eng_text = "\n".join([f"• {x['report_date']}: {x['reason_for_recall'][:150]}..." for x in r['results']])
            try:
                model = genai.GenerativeModel("gemini-2.0-flash")
                return model.generate_content(f"Translate FDA recall reasons to Korean naturally:\n{eng_text}").text
            except Exception: return eng_text
        return "✅ FDA 리콜 이력 없음 (CLEAN)"
    except Exception: return "ℹ️ FDA 데이터 없음"

def run_deep_analysis(ticker, price, score, indicators, fda, earnings):
    warn = f"🚨실적발표 {earnings['d_day']} 전!" if earnings['diff'] <= 7 else ""
    # 1) Tavily로 최신 뉴스/재료 검색
    news = tavily_search(f"{ticker} stock latest news catalyst", want_answer=True, max_results=5)
    web_context = (news["answer"] + "\n" + news["snippets"]).strip() or "(검색된 뉴스 없음)"
    # 2) 검색 결과를 컨텍스트로 주입해 OpenRouter 추론
    # FDA는 의료/바이오 종목일 때만 프롬프트에 포함 (비의료주는 제외)
    fda_line = f", FDA: {fda}" if fda else ""
    prompt = f"""
    [ROLE] 한국의 주식 고수 (반말, 면책조항 절대 금지)
    [TARGET] {ticker} (${price})
    [QUANT] Score: {score}, {indicators['trend']}, {indicators['whale']}, {indicators['squeeze']}
    [DATA] 실적: {earnings['date']} ({earnings['d_day']}) {warn}{fda_line}
    [최신 웹검색 결과]
    {web_context}
    [MISSION]
    위 웹검색 결과를 근거로 분석해라. 한국 주식 용어 사용.
    [OUTPUT]
    ## ⚡ 뉴스 & 팩트체크
    (3줄 요약)
    ## ⚠️ 리스크 진단
    (핵심 위험요소)
    ## 🏛️ 최종 대응 전략
    (풀매수/분할매수/관망/손절) - (이유 한줄)
    """
    return openrouter_chat(prompt, temperature=0.3, timeout=30)

# ==========================================
# [4] 메인 로직 (버튼 토글 기능 적용)
# ==========================================
st.markdown(f"<div class='macro-bar'>{get_macro_ticker()}</div>", unsafe_allow_html=True)

# Session State 초기화 (분석 상태 관리)
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

def toggle_analysis():
    st.session_state.is_running = not st.session_state.is_running

c1, c2 = st.columns([3, 1])
ticker = c1.text_input("TICKER", value="RKLB", label_visibility="collapsed").upper().strip()

# 버튼 라벨 및 콜백 설정
btn_label = "🛑 분석 중단" if st.session_state.is_running else "🔥 분석 시작"
# 버튼을 누르면 toggle_analysis 함수가 실행되어 상태가 반전됨
c2.button(btn_label, on_click=toggle_analysis)

# 상태가 True일 때만 분석 실행
if st.session_state.is_running:
    with st.spinner(f"AI 퀀트 엔진: {ticker} 실시간 분석 중..."):
        try:
            client = RESTClient(API_KEY)
            end = datetime.now(pytz.timezone("America/New_York"))
            start = end - timedelta(days=150) 
            aggs = list(client.list_aggs(ticker, 1, "day", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), limit=50000))
            
            if not aggs:
                st.error("데이터를 찾을 수 없습니다. 티커를 확인해주세요.")
                st.session_state.is_running = False # 데이터 없으면 중단
            else:
                df = pd.DataFrame(aggs)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.rename(columns={'open':'open','high':'high','low':'low','close':'close','volume':'volume'})
                
                df = calculate_quant_metrics(df)
                if len(df) < 20:
                    st.error("데이터 부족")
                    st.session_state.is_running = False
                    st.stop()

                row = df.iloc[-1]
                info = get_ticker_details(ticker, client)
                earnings = get_earnings_schedule(ticker)
                fda_data = get_fda_data(info['name']) if info['is_bio'] else ""
                
                score = get_ai_score(row)
                grade = "S (강력매수)" if score >= 80 else "A (매수)" if score >= 60 else "B (중립)" if score >= 40 else "C (매도)"
                score_col = "#ff4757" if score >= 60 else "#f1c40f" if score >= 40 else "#00a8ff"
                
                target = row['close'] + (row['ATR'] * 2)
                cut = row['close'] - (row['ATR'] * 1.5)
                
                is_up = row['close'] > row['SMA20']
                trend = "📈 상승세" if is_up else "📉 하락세"
                trend_col = "#ff4757" if is_up else "#00a8ff"
                
                whale_ratio = row['volume'] / max(row['VolAvg20'], 1)
                whale = f"🐋 고래출현 ({whale_ratio:.1f}x)" if whale_ratio > 3.0 else "일반 수급"
                is_squeeze = row['Bandwidth'] < 0.10
                squeeze_msg = "⚡ 스퀴즈 (폭발 임박)" if is_squeeze else "일반"
                
                # UI Render
                st.markdown(f"<h1 style='margin:0'>{ticker}</h1>", unsafe_allow_html=True)
                if earnings['diff'] <= 7:
                    st.markdown(f"<div style='text-align:center'><span class='earnings-badge'>🚨 실적 {earnings['d_day']}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='color:#fff'>${row['close']:.2f}</h2>", unsafe_allow_html=True)
                
                st.markdown(f"<div class='big-score' style='color:{score_col}; text-shadow: 0 0 20px {score_col}'>{score}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center'><span class='grade-badge' style='border: 2px solid {score_col}; color:{score_col}'>{grade}</span></div>", unsafe_allow_html=True)

                st.plotly_chart(draw_chart_k_style(df, ticker), use_container_width=True)
                
                c_1, c_2, c_3 = st.columns(3)

                # 1. 추세 (Trend) - 상승:네온레드 / 하락:네온블루
                if is_up:
                    trend_color = "#ff003c" # Neon Red
                    trend_desc = "강한 상승 추세"
                else:
                    trend_color = "#00f2ff" # Cyan Blue
                    trend_desc = "약한 하락 추세"

                with c_1:
                    st.markdown(f"""
                    <div class='neon-card' style='color: {trend_color};'>
                        <div class='metric-title'>TREND</div>
                        <div class='metric-value' style='text-shadow: 0 0 10px {trend_color}'>{trend}</div>
                        <div class='neon-desc'>{trend_desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # 2. RSI - 네온 옐로우 (상태 표시)
                rsi_color = "#ffe600" # Neon Yellow
                if row['RSI'] < 30:
                    rsi_desc = "과매도 (매수신호)"
                elif row['RSI'] > 70:
                    rsi_desc = "과매수 (매도신호)"
                else:
                    rsi_desc = "중립 (관망)"

                with c_2:
                    st.markdown(f"""
                    <div class='neon-card' style='color: {rsi_color};'>
                        <div class='metric-title'>RSI (14)</div>
                        <div class='metric-value' style='text-shadow: 0 0 10px {rsi_color}'>{row['RSI']:.1f}</div>
                        <div class='neon-desc'>{rsi_desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # 3. 거래량 (Volume) - 네온 그린 (고래 출현시 더 밝게)
                vol_color = "#39ff14" if "고래" in whale else "#008000" # Neon Green vs Dark Green
                vol_desc = "고래 활동 감지!" if "고래" in whale else "정상 거래량"

                with c_3:
                    st.markdown(f"""
                    <div class='neon-card' style='color: {vol_color};'>
                        <div class='metric-title'>VOLUME</div>
                        <div class='metric-value' style='text-shadow: 0 0 10px {vol_color}'>{whale}</div>
                        <div class='neon-desc'>{vol_desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # ==========================================
                # [네온 스타일] 특이 신호 감지 (항상 노출)
                # ==========================================
                has_signal = bool(is_squeeze or (whale_ratio >= 3.0))

                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)  # 간격

                if has_signal:
                    # 켜진 네온 박스
                    html_content = "<div class='signal-box-on'>"
                    html_content += "<div style='font-weight: bold; font-size: 1.1rem; margin-bottom: 10px;'>🚨 SIGNAL DETECTED</div>"
                    if is_squeeze: html_content += "<div>⚡ <b>BOLLINGER SQUEEZE</b> (에너지 응축)</div>"
                    if whale_ratio >= 3.0: html_content += f"<div style='margin-top:8px'>🟣 <b>WHALE VOLUME</b> (평소의 {whale_ratio:.1f}배)</div>"
                    html_content += "</div>"
                    st.markdown(html_content, unsafe_allow_html=True)
                else:
                    # 꺼진 네온 박스
                    st.markdown(f"""
                    <div class='signal-box-off'>
                        <div style='font-size: 1.2rem; margin-bottom:5px;'>✅ SYSTEM NORMAL</div>
                        <div style='font-size: 0.8rem;'>특이 신호 감지되지 않음</div>
                    </div>
                    """, unsafe_allow_html=True)

                c_t, c_s = st.columns(2)
                with c_t: st.markdown(f"<div class='target-box'><div>Target</div><div style='font-size:1.4rem'>${target:.2f}</div></div>", unsafe_allow_html=True)
                with c_s: st.markdown(f"<div class='stop-box'><div>Cut</div><div style='font-size:1.4rem'>${cut:.2f}</div></div>", unsafe_allow_html=True)

                st.divider()
                st.markdown("### 🧬 AI 심층 분석")
                
                ind_dict = {"trend": trend, "whale": whale, "squeeze": squeeze_msg}
                report = run_deep_analysis(ticker, row['close'], score, ind_dict, fda_data, earnings)
                st.markdown(report)
                if info['is_bio']:
                    with st.expander("💊 FDA 리콜 정보", expanded=False): st.write(fda_data)

                st.session_state.last_analysis = {
                    "ticker": ticker, "price": f"${row['close']:.2f}", "score": score, "grade": grade,
                    "trend": trend, "report": report
                }

        except Exception as e:
            st.error(f"오류: {e}")
            st.session_state.is_running = False

# ==========================================
# [5] 채팅 (추천 기능 포함)
# ==========================================
st.divider()
if q := st.chat_input("종목 추천이나 분석 내용에 대해 질문하세요..."):
    with st.chat_message("user"):
        # 입력된 텍스트는 CSS로 검정색으로 보임
        st.write(q)
    with st.chat_message("assistant"):
        with st.spinner("AI가 생각 중입니다..."):
            try:
                if hasattr(st.session_state, 'last_analysis') and "추천" not in q and "종목" not in q:
                    # 분석 후속 질문 -> 웹검색 불필요, OpenRouter 추론만
                    analysis = st.session_state.last_analysis
                    context = f"[종목] {analysis['ticker']} ({analysis['price']}) / [점수] {analysis['score']}\n[분석] {analysis['report']}\n[질문] {q}"
                    content = f"{context}\n\n위 내용을 바탕으로 답변해. (한국 주식 고수 말투, 반말)"
                else:
                    # 종목 추천 / 시장 질문 -> Tavily 실시간 검색 후 주입
                    today = datetime.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d")
                    web = tavily_search(f"hottest US stocks today high volume catalyst {today}", want_answer=True, max_results=5)
                    web_context = (web["answer"] + "\n" + web["snippets"]).strip() or "(검색 결과 없음)"
                    content = f"""
[Context] 오늘은 {today}이다. 사용자가 주식 종목을 추천해달라고 하거나 시장 상황을 묻는다.
[최신 웹검색 결과]
{web_context}
[Mission]
1. 위 검색 결과를 근거로 현재 미국 시장에서 핫한 섹터와 종목 3개를 추천해라.
2. 이유를 명확히 설명해라 (거래량, 호재, 뉴스 등).
3. 한국 주식 은어 사용 가능 (떡상, 줍줍 등), 단 '투자 책임' 언급은 금지.
[질문] {q}
"""
                st.write(openrouter_chat(content, temperature=0.5, timeout=30))
            except Exception as e:
                st.error(f"채팅 오류: {e}")

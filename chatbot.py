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
import time

# ==========================================
# [1] UI: 사이버펑크 퀀트 스타일
# ==========================================
st.set_page_config(
    page_title="QUANTUM AI TERMINAL",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    /* 전체 테마: 블랙 & 네온 */
    .stApp { background-color: #050505; color: #e0e0e0; }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 폰트 & 타이포그래피 */
    h1 { font-family: 'Courier New', monospace; color: #fff; text-align: center; }
    h2, h3 { font-family: 'Courier New', monospace; color: #FFD700 !important; text-align: center; }
    
    /* 점수판 */
    .big-score {
        font-size: 5rem; font-weight: 900; color: #00ff41; 
        text-align: center; text-shadow: 0 0 20px rgba(0, 255, 65, 0.5);
        line-height: 1.1; margin-top: 10px;
    }
    .grade-badge {
        font-size: 1.5rem; font-weight: bold; padding: 5px 15px;
        border: 2px solid #00ff41; border-radius: 5px; color: #00ff41;
        display: inline-block; margin-bottom: 20px;
    }

    /* 카드 디자인 */
    .signal-card {
        background-color: #111; border: 1px solid #333; border-radius: 8px;
        padding: 15px; margin-bottom: 15px;
    }
    .metric-title { font-size: 0.9rem; color: #aaa; font-weight: bold; } 
    .metric-value { font-size: 1.2rem; font-weight: bold; color: #fff; margin-top: 5px;}
    
    /* 동적 진단 박스 스타일 (NEW) */
    .diagnosis-box {
        background-color: #111; 
        border-left: 5px solid #555;
        padding: 15px; 
        margin-bottom: 8px; 
        font-size: 1rem;
        border-radius: 0 8px 8px 0;
    }
    .diag-title { font-weight: bold; margin-right: 10px; }
    
    /* 매크로 바 */
    .macro-bar {
        background-color: #0a0a0a; border-bottom: 1px solid #333;
        padding: 8px; text-align: center; font-size: 0.9rem; color: #ff9f43;
        font-family: 'Courier New', monospace; margin-bottom: 20px; font-weight: bold;
    }
    
    /* 가격 타겟 박스 */
    .target-box { border: 1px solid #00ff41; color: #00ff41; padding: 10px; border-radius: 5px; text-align: center; background: rgba(0, 255, 65, 0.05); }
    .stop-box { border: 1px solid #ff4b4b; color: #ff4b4b; padding: 10px; border-radius: 5px; text-align: center; background: rgba(255, 75, 75, 0.05); }

    /* 실적 배지 */
    .earnings-badge { background-color: #ff4757; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; }

    /* 버튼 */
    .stButton > button {
        width: 100%; background-color: #003300; color: #00ff41;
        border: 1px solid #00ff41; height: 3.5em; font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover { background-color: #00ff41; color: black; box-shadow: 0 0 15px #00ff41; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# [2] API 키 로드
# ==========================================
try:
    API_KEY = st.secrets["POLYGON_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
    FDA_API_KEY = st.secrets["FDA_API_KEY"]
except:
    st.error("🚨 API 키 오류: secrets.toml 파일을 확인하세요.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# [3] 퀀트 엔진
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
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    return df.iloc[-1]

def get_ai_score(row):
    score = 50
    if row['close'] > row['SMA20']: score += 15
    else: score -= 10
    
    if 50 <= row['RSI'] <= 70: score += 15
    elif row['RSI'] > 75: score -= 5
    elif row['RSI'] < 30: score += 20
    
    if row['MACD'] > row['Signal']: score += 15
    
    vol_ratio = row['volume'] / row['VolAvg20']
    if vol_ratio > 3.0: score += 20
    elif vol_ratio > 1.5: score += 10
    
    return min(100, max(0, int(score)))

def get_macro_ticker():
    try:
        data = yf.download(['^TNX', '^VIX', 'CL=F', 'GC=F'], period='1d', progress=False)['Close'].iloc[-1]
        tnx = data['^TNX'].item() if hasattr(data['^TNX'], 'item') else data['^TNX']
        vix = data['^VIX'].item() if hasattr(data['^VIX'], 'item') else data['^VIX']
        return f"국채10년: {tnx:.2f}% | VIX: {vix:.2f} | 유가: ${data['CL=F']:.1f} | 금: ${data['GC=F']:.0f}"
    except: return "매크로 데이터 로딩 중..."

# ==========================================
# [4] 인텔리전스 엔진
# ==========================================
@st.cache_data
def get_ticker_details(ticker, _client):
    try:
        d = _client.get_ticker_details(ticker)
        ind = getattr(d, "sic_description", "").upper()
        name = d.name
        is_bio = any(x in ind+name.upper() for x in ["PHARMA", "BIO", "DRUG", "MED", "LIFE"])
        return {"name": name, "is_bio": is_bio}
    except: return {"name": ticker, "is_bio": False}

@st.cache_data
def get_earnings_schedule(ticker):
    try:
        stock = yf.Ticker(ticker)
        try:
            cal = stock.calendar
            if cal and 'Earnings Date' in cal: return calc_d_day(cal['Earnings Date'][0])
        except: pass
        try:
            df = stock.get_earnings_dates(limit=10)
            future = df[df.index > datetime.now()].sort_index()
            if not future.empty: return calc_d_day(future.index[0])
        except: pass
    except: pass
    
    try:
        url = "https://api.perplexity.ai/chat/completions"
        h = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
        msg = [{"role":"user", "content":f"Find next earnings date for {ticker}. Output YYYY-MM-DD only."}]
        res = requests.post(url, json={"model":"sonar","messages":msg,"temperature":0.1}, headers=h, timeout=4)
        match = re.search(r'\d{4}-\d{2}-\d{2}', res.json()['choices'][0]['message']['content'])
        if match: return calc_d_day(datetime.strptime(match.group(0), "%Y-%m-%d").date())
    except: pass
    
    return {"d_day": "-", "date": "미정", "diff": 999}

def calc_d_day(date_obj):
    if isinstance(date_obj, datetime): date_obj = date_obj.date()
    diff = (date_obj - datetime.now().date()).days
    d_day = "D-Day" if diff == 0 else f"D-{diff}" if diff > 0 else "완료"
    return {"d_day": d_day, "date": date_obj.strftime("%Y-%m-%d"), "diff": diff}

def get_fda_data(name):
    clean = re.sub(r'[,.]|Inc|Corp|Ltd', '', name).strip().replace(" ", "+")
    url = f"https://api.fda.gov/drug/enforcement.json?api_key={FDA_API_KEY}&search=openfda.manufacturer_name:{clean}&limit=3&sort=report_date:desc"
    try:
        r = requests.get(url, timeout=3).json()
        if 'results' in r:
            eng_text = "\n".join([f"• {x['report_date']}: {x['reason_for_recall'][:150]}..." for x in r['results']])
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                trans = model.generate_content(f"Translate FDA recall reasons to Korean naturally:\n{eng_text}").text
                return trans
            except: return eng_text
        return "✅ FDA 리콜 이력 없음 (CLEAN)"
    except: return "ℹ️ FDA 데이터 없음"

def run_deep_analysis(ticker, price, score, indicators, news_data, fda, earnings):
    mode = "바이오" if fda and "FDA" in fda else "기술주"
    warn = f"🚨실적발표 {earnings['d_day']} 전!" if earnings['diff'] <= 7 else ""
    
    prompt = f"""
    [ROLE] 월스트리트 퀀트 펀드매니저
    [TARGET] {ticker} (현재가: ${price})
    [QUANT] Score: {score}, 추세: {indicators['trend']}, 수급: {indicators['whale']}
    [DATA] 실적: {earnings['date']} ({earnings['d_day']}) {warn}, FDA: {fda}
    [MISSION] 실시간 뉴스(24h) 결합 분석. 면책조항 금지.
    [OUTPUT]
    ## ⚡ 뉴스 & 팩트
    (핵심만)
    ## ⚠️ 리스크 분석
    (실적, FDA, 수급 악재 등)
    ## 🏛️ 최종 전략
    (매수/관망/매도) - (이유 한줄)
    """
    url = "https://api.perplexity.ai/chat/completions"
    h = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    try:
        return requests.post(url, json={"model":"sonar","messages":[{"role":"user","content":prompt}],"temperature":0.2}, headers=h).json()['choices'][0]['message']['content']
    except: return "AI 분석 연결 실패"

# ==========================================
# [5] 메인 애플리케이션
# ==========================================
st.markdown(f"<div class='macro-bar'>{get_macro_ticker()}</div>", unsafe_allow_html=True)

c1, c2 = st.columns([3, 1])
ticker = c1.text_input("TICKER", value="IONQ", label_visibility="collapsed").upper().strip()
run = c2.button("시스템 스캔 시작 🚀")

if run:
    with st.spinner("AI 퀀트 엔진: 데이터 수집 및 분석 중..."):
        try:
            client = RESTClient(API_KEY)
            end = datetime.now(pytz.timezone("America/New_York"))
            start = end - timedelta(days=60)
            aggs = list(client.list_aggs(ticker, 1, "day", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), limit=50000))
            
            if not aggs:
                st.error("데이터를 찾을 수 없습니다.")
            else:
                df = pd.DataFrame(aggs)
                df = df.rename(columns={'open':'open','high':'high','low':'low','close':'close','volume':'volume'})
                row = calculate_quant_metrics(df)
                
                info = get_ticker_details(ticker, client)
                earnings = get_earnings_schedule(ticker)
                fda_data = get_fda_data(info['name']) if info['is_bio'] else ""
                
                score = get_ai_score(row)
                grade = "S (강력매수)" if score >= 80 else "A (매수)" if score >= 60 else "B (중립)" if score >= 40 else "C (매도)"
                score_col = "#00ff41" if score >= 60 else "#f1c40f" if score >= 40 else "#ff4757"
                
                target = row['close'] + (row['ATR'] * 2)
                cut = row['close'] - (row['ATR'] * 1.5)
                
                trend = "📈 상승세" if row['close'] > row['SMA20'] else "📉 하락세"
                whale_ratio = row['volume']/row['VolAvg20']
                whale = f"🐋 고래출현 ({whale_ratio:.1f}x)" if whale_ratio > 3.0 else "일반 수급"
                
                # UI 출력
                st.markdown(f"<h1 style='margin:0'>{ticker}</h1>", unsafe_allow_html=True)
                if earnings['diff'] <= 7:
                    st.markdown(f"<div style='text-align:center'><span class='earnings-badge'>🚨 실적 {earnings['d_day']}</span></div>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='color:#fff'>${row['close']:.2f}</h2>", unsafe_allow_html=True)
                
                st.markdown(f"<div class='big-score' style='color:{score_col}'>{score}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center'><span class='grade-badge' style='border-color:{score_col}; color:{score_col}'>{grade}</span></div>", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"""<div class='signal-card'><div class='metric-title'>추세 (TREND)</div><div class='metric-value' style='color:{'#00ff41' if '상승' in trend else '#ff4757'}'>{trend}</div></div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class='signal-card'><div class='metric-title'>RSI (14)</div><div class='metric-value'>{row['RSI']:.1f}</div></div>""", unsafe_allow_html=True)
                with c3:
                    wh_col = "#d63031" if "일반" in whale else "#a29bfe"
                    st.markdown(f"""<div class='signal-card'><div class='metric-title'>거래량 (VOLUME)</div><div class='metric-value' style='color:{wh_col}'>{whale}</div></div>""", unsafe_allow_html=True)

                # ==========================================
                # [🔥 실시간 맞춤형 진단 기능]
                # ==========================================
                with st.expander("🔍 현재 상태 정밀 진단 (Click)", expanded=True):
                    # 1. RSI 진단
                    rsi_val = row['RSI']
                    if rsi_val <= 30:
                        rsi_msg = f"현재 {rsi_val:.1f} (30 이하) ➔ 🟢 **과매도 구간! 기술적 반등 가능성 높음 (줍줍 기회)**"
                        rsi_col = "#00ff41"
                    elif rsi_val >= 70:
                        rsi_msg = f"현재 {rsi_val:.1f} (70 이상) ➔ 🔴 **과매수 구간! 단기 조정 가능성 있음 (주의)**"
                        rsi_col = "#ff4757"
                    else:
                        rsi_msg = f"현재 {rsi_val:.1f} ➔ ⚪ **중립 구간 (일반적인 흐름)**"
                        rsi_col = "#ccc"
                    
                    # 2. 거래량 진단
                    if whale_ratio >= 3.0:
                        vol_msg = f"평소의 {whale_ratio:.1f}배 ➔ 🟣 **고래(세력)가 물량을 쓸어담거나 던지는 중! 방향성 주목**"
                        vol_col = "#a29bfe"
                    elif whale_ratio >= 1.5:
                        vol_msg = f"평소의 {whale_ratio:.1f}배 ➔ 🟡 **거래량이 살아나는 중**"
                        vol_col = "#f1c40f"
                    else:
                        vol_msg = f"평소와 비슷함 ({whale_ratio:.1f}배) ➔ ⚪ **특이사항 없음**"
                        vol_col = "#ccc"

                    # 3. 추세 진단
                    if row['close'] > row['SMA20']:
                        trend_msg = "현재가 > 20일선 ➔ 🟢 **상승 추세가 살아있음 (보유 관점)**"
                        trend_col = "#00ff41"
                    else:
                        trend_msg = "현재가 < 20일선 ➔ 🔴 **하락 추세 진행 중 (방어적 대응 필요)**"
                        trend_col = "#ff4757"

                    st.markdown(f"""
                    <div class='diagnosis-box' style='border-left-color: {rsi_col}'><span class='diag-title' style='color:{rsi_col}'>RSI 진단:</span> {rsi_msg}</div>
                    <div class='diagnosis-box' style='border-left-color: {vol_col}'><span class='diag-title' style='color:{vol_col}'>수급 진단:</span> {vol_msg}</div>
                    <div class='diagnosis-box' style='border-left-color: {trend_col}'><span class='diag-title' style='color:{trend_col}'>추세 진단:</span> {trend_msg}</div>
                    """, unsafe_allow_html=True)

                c_t, c_s = st.columns(2)
                with c_t:
                    st.markdown(f"<div class='target-box'><div>1차 목표가 (Target)</div><div style='font-size:1.4rem; font-weight:bold'>${target:.2f}</div></div>", unsafe_allow_html=True)
                with c_s:
                    st.markdown(f"<div class='stop-box'><div>1차 손절가 (Cut)</div><div style='font-size:1.4rem; font-weight:bold'>${cut:.2f}</div></div>", unsafe_allow_html=True)

                st.divider()
                st.markdown("### 🧬 AI 심층 분석 리포트")
                
                indicators = {"trend": trend, "whale": whale}
                with st.spinner("AI 리포트 작성 중..."):
                    report = run_deep_analysis(ticker, row['close'], score, indicators, "", fda_data, earnings)
                    st.markdown(report)
                    
                    if info['is_bio']:
                        with st.expander("💊 FDA 리콜 데이터 (한글 번역본)", expanded=False):
                            st.write(fda_data)

        except Exception as e:
            st.error(f"시스템 오류 발생: {e}")

st.divider()
if q := st.chat_input("AI 애널리스트에게 질문하기..."):
    with st.chat_message("user"): st.write(q)
    with st.chat_message("assistant"):
        with st.spinner("검색 중..."):
            try:
                url = "https://api.perplexity.ai/chat/completions"
                h = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
                res = requests.post(url, json={"model":"sonar","messages":[{"role":"user","content":f"질문: {q} (주식관련, 짧게, 면책조항X)"}],"temperature":0.1}, headers=h).json()
                st.write(res['choices'][0]['message']['content'])
            except: st.error("채팅 오류")

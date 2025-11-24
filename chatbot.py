import streamlit as st
from polygon import RESTClient
from datetime import datetime, timedelta
from collections import defaultdict
import google.generativeai as genai
import pytz
import requests
import re
import yfinance as yf # 👈 실적 발표일 조회를 위해 추가

# ==========================================
# [1] UI 설정
# ==========================================
st.set_page_config(
    page_title="세력 탐지기 Ultimate",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding: 1rem 1rem 5rem 1rem !important;}
    .stButton > button {width: 100%; border-radius: 12px; height: 3em; font-weight: bold;}
    div[data-testid="stMetric"] {background-color: #f0f2f6; padding: 10px; border-radius: 10px; text-align: center;}
    /* D-Day 뱃지 스타일 */
    .d-day-badge {
        background-color: #ff4b4b; color: white; padding: 2px 8px; border-radius: 5px; font-weight: bold; font-size: 0.8em;
    }
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
    st.error("🚨 API 키 설정 필요 (.streamlit/secrets.toml)")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# [3] 스마트 함수들
# ==========================================
@st.cache_data
def get_earnings_info(ticker):
    """Yahoo Finance에서 다음 실적 발표일 조회 및 D-Day 계산"""
    try:
        stock = yf.Ticker(ticker)
        # 캘린더 데이터 가져오기
        calendar = stock.calendar
        
        earnings_date = None
        # 데이터 구조가 버전에 따라 다를 수 있어 처리
        if isinstance(calendar, dict) and 'Earnings Date' in calendar:
             earnings_date = calendar['Earnings Date'][0]
        elif hasattr(calendar, 'iloc'): # DataFrame인 경우
             earnings_date = calendar.iloc[0][0]
        
        if earnings_date:
            today = datetime.now().date()
            e_date = earnings_date.date()
            days_left = (e_date - today).days
            
            # D-Day 문자열 포맷팅
            if days_left == 0: d_str = "D-Day (오늘)"
            elif days_left > 0: d_str = f"D-{days_left}"
            else: d_str = "발표 완료"
            
            return {
                "date": e_date.strftime("%Y-%m-%d"),
                "d_day": d_str,
                "days_left": days_left
            }
        return {"date": "미정", "d_day": "-", "days_left": 999}
    except:
        return {"date": "정보 없음", "d_day": "-", "days_left": 999}

@st.cache_data
def get_ticker_info(ticker, _client):
    try:
        details = _client.get_ticker_details(ticker)
        name = details.name
        industry = getattr(details, "sic_description", "").upper()
        bio_keywords = ["PHARMA", "BIO", "DRUG", "MEDICAL", "SURGICAL", "LIFE", "HEALTH", "THERAP"]
        is_bio = any(k in industry for k in bio_keywords) or any(k in name.upper() for k in bio_keywords)
        return {"name": name, "industry": industry if industry else "Unknown", "is_bio": is_bio}
    except:
        return {"name": ticker, "industry": "Unknown", "is_bio": False}

def get_clean_name(name):
    name = re.sub(r'[,.]', '', name)
    remove = ['Inc', 'Corp', 'Corporation', 'Ltd', 'PLC', 'Group', 'Holdings', 'Therapeutics', 'Pharma']
    for word in remove:
        name = re.sub(r'\b' + word + r'\b', '', name, flags=re.IGNORECASE)
    return name.strip()

def get_fda_data(company_name):
    clean_name = get_clean_name(company_name)
    query = clean_name.replace(" ", "+")
    url = f"https://api.fda.gov/drug/enforcement.json?api_key={FDA_API_KEY}&search=openfda.manufacturer_name:{query}&limit=3&sort=report_date:desc"
    try:
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            results = res.json().get('results', [])
            if results:
                summary = []
                for r in results:
                    summary.append(f"• {r.get('report_date','-')} ({r.get('status','-')})\n  └ {r.get('reason_for_recall','')[:60]}...")
                return "\n".join(summary)
            return "✅ 최근 리콜 없음"
        return "ℹ️ FDA 데이터 없음"
    except: return "❌ FDA 연결 실패"

def run_ai_analysis(mode, system_data, fda_data, earnings_data):
    """실적 발표일(earnings_data)을 프롬프트에 추가"""
    
    # 실적 발표 임박 시 경고 추가
    earnings_warning = ""
    if earnings_data['days_left'] <= 7 and earnings_data['days_left'] >= 0:
        earnings_warning = f"\n🚨 [긴급] 실적 발표가 {earnings_data['d_day']} 남았습니다! 변동성 주의 경고를 포함하세요."

    if mode == "BIO":
        role = "바이오/제약 전문 투자자"
        prompt = f"""
        [데이터]
        {system_data}
        [실적일정]
        다음 발표일: {earnings_data['date']} ({earnings_data['d_day']}) {earnings_warning}
        [FDA/임상]
        {fda_data}
        
        [지시]
        1. FDA 이슈와 실적 일정(Earnings)을 고려해 리스크 분석.
        2. 최신 임상 결과 및 뉴스 검색.
        3. 실적 발표가 가까우면 관망 권고 고려.
        
        [양식]
        ## 💊 FDA/임상/실적
        (내용)
        ## 📰 뉴스 팩트체크
        (내용)
        ## 🎯 결론
        (매수🟢/관망🟡/매도🔴) - (이유)
        """
    else:
        role = "월스트리트 기술주 전문가"
        prompt = f"""
        [데이터]
        {system_data}
        [실적일정]
        다음 발표일: {earnings_data['date']} ({earnings_data['d_day']}) {earnings_warning}
        
        [지시]
        1. 실적 발표 일정에 따른 변동성 리스크 분석.
        2. 최근 24시간 내 공시 및 뉴스 검색.
        3. 기술적 위치 분석.
        
        [양식]
        ## 🏢 실적/뉴스 이슈
        (내용)
        ## ⚠️ 리스크 체크
        (내용)
        ## 🎯 결론
        (매수🟢/관망🟡/매도🔴) - (이유)
        """

    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "sonar",
        "messages": [{"role": "system", "content": f"당신은 {role}입니다. 면책조항 금지. 팩트 기반 직설적 답변."}, {"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    try:
        return requests.post(url, json=payload, headers=headers).json()["choices"][0]["message"]["content"]
    except Exception as e: return f"AI 분석 실패: {e}"

def extract_signal(text):
    text = text.lower()
    if "🟢" in text or "매수" in text: return "매수 기회", "#d4edda", "#155724"
    elif "🔴" in text or "매도" in text: return "위험/매도", "#f8d7da", "#721c24"
    else: return "관망 필요", "#fff3cd", "#856404"

# ==========================================
# [4] 메인 로직
# ==========================================
if "chat_history" not in st.session_state: st.session_state.chat_history = []

st.title("📡 미국 주식 세력 탐지기")
st.caption("Bio/Tech Auto-Detect + Earnings Alert 📅")

col_in, col_btn = st.columns([2, 1])
ticker = col_in.text_input("티커", value="NVDA", label_visibility="collapsed").upper().strip()
run = col_btn.button("분석 🚀", type="primary", use_container_width=True)

if run:
    with st.spinner(f"[{ticker}] 데이터 채굴 및 실적 일정 조회 중..."):
        try:
            client = RESTClient(API_KEY)
            
            # 1. 정보 수집 (기본정보 + 실적발표일)
            info = get_ticker_info(ticker, client)
            earnings = get_earnings_info(ticker) # 👈 실적 조회 추가됨
            
            company_name = info['name']
            is_bio = info['is_bio']
            mode = "BIO" if is_bio else "GENERAL"

            # 2. 차트 데이터
            end_dt = datetime.now(pytz.timezone("America/New_York"))
            start_dt = end_dt - timedelta(days=14)
            aggs = list(client.list_aggs(ticker, 1, "minute", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), limit=50000))

            if not aggs:
                st.error(f"❌ '{ticker}' 데이터 없음")
            else:
                current_price = aggs[-1].close
                total_vol = sum(a.volume for a in aggs)
                total_pv = sum(((a.high+a.low+a.close)/3)*a.volume for a in aggs)
                vwap = total_pv/total_vol if total_vol else 0
                price_vol = defaultdict(int)
                for a in aggs: price_vol[round(a.close, 1)] += a.volume
                support = max(price_vol, key=price_vol.get)
                diff = ((current_price - vwap)/vwap)*100

                fda_info = get_fda_data(company_name) if is_bio else "해당 없음"

                st.session_state.analysis_data = {
                    "ticker": ticker, "name": company_name, "price": current_price, "mode": mode
                }

                # 3. 화면 표시 (배지)
                badge_bg = "#e6fffa" if is_bio else "#e6f7ff"
                badge_txt = "🧬 BIO" if is_bio else "💻 TECH"
                
                # 실적 D-Day에 따른 경고 배지
                earnings_badge = ""
                if earnings['days_left'] <= 7 and earnings['days_left'] >= 0:
                     earnings_badge = f"<span class='d-day-badge'>🚨 실적 {earnings['d_day']}</span>"
                
                st.markdown(f"""
                <div style='text-align:center; margin-bottom:10px;'>
                    <span style='background-color:{badge_bg}; padding:5px 10px; border-radius:5px; font-weight:bold; color:#555; margin-right:5px;'>{badge_txt}</span>
                    {earnings_badge}
                </div>
                """, unsafe_allow_html=True)

                # 메트릭 (2열 -> 2열 2행으로 확장)
                c1, c2 = st.columns(2)
                c1.metric("현재가", f"${current_price}")
                c2.metric("세력평단", f"${vwap:.2f}", f"{diff:.1f}%")
                
                c3, c4 = st.columns(2)
                c3.metric("강력 지지선", f"${support}")
                c4.metric("다음 실적발표", f"{earnings['date']}", f"{earnings['d_day']}") # 👈 실적 메트릭 추가

                # 4. AI 분석
                sys_data = f"종목: {ticker}({company_name}), 가격: {current_price}, VWAP: {vwap:.2f}"
                ai_res = run_ai_analysis(mode, sys_data, fda_info, earnings) # 👈 실적 정보 AI 전달

                # 결과 카드
                sig_text, bg, txt = extract_signal(ai_res)
                st.markdown(f"""
                <div style="background-color:{bg}; padding:15px; border-radius:12px; text-align:center; margin:15px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    <h3 style="color:{txt}; margin:0; font-size:1.5rem;">{sig_text}</h3>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("📊 상세 분석 결과", expanded=False):
                    st.markdown(ai_res)
                
                if is_bio:
                    with st.expander("💊 FDA 리콜 내역", expanded=False): st.text(fda_info)

                st.session_state.chat_history.append({"role": "assistant", "content": f"[{ticker}] {sig_text}\n{ai_res}"})

        except Exception as e:
            st.error(f"오류: {e}")

# ==========================================
# [5] 채팅
# ==========================================
st.divider()
st.subheader("💬 AI 질문")

msgs = st.session_state.chat_history[-2:] if len(st.session_state.chat_history) > 2 else st.session_state.chat_history
for msg in msgs:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if q := st.chat_input("질문 (예: 실적 전망 어때?)"):
    st.session_state.chat_history.append({"role": "user", "content": q})
    with st.chat_message("user"): st.write(q)
    
    with st.chat_message("assistant"):
        with st.spinner("검색 중..."):
            ctx = ""
            if st.session_state.analysis_data:
                d = st.session_state.analysis_data
                ctx = f"[종목:{d['ticker']}, 모드:{d['mode']}]"
            
            p = f"데이터:{ctx}\n질문:{q}\n지시: 최신뉴스,실적전망포함,면책조항금지."
            h = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
            d = {"model": "sonar", "messages": [{"role": "user", "content": p}], "temperature": 0.2}
            try:
                r = requests.post("https://api.perplexity.ai/chat/completions", json=d, headers=h).json()
                ans = r["choices"][0]["message"]["content"]
                st.write(ans)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
            except: st.error("오류")

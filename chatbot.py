import streamlit as st
from polygon import RESTClient
from datetime import datetime, timedelta
from collections import defaultdict
import google.generativeai as genai
import pytz
import requests
import re
import yfinance as yf

# ==========================================
# [1] UI 및 모바일 최적화 설정
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
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        font-weight: bold;
        font-size: 1rem;
    }
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .d-day-badge {
        background-color: #ff4b4b; 
        color: white; 
        padding: 3px 8px; 
        border-radius: 6px; 
        font-size: 0.8rem; 
        font-weight: bold;
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
except (FileNotFoundError, KeyError):
    st.error("🚨 API 키 설정 오류! secrets.toml 파일을 확인하세요.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# [3] 핵심 유틸리티 함수
# ==========================================

@st.cache_data
def get_ticker_info(ticker, _client):
    try:
        details = _client.get_ticker_details(ticker)
        name = details.name
        industry = getattr(details, "sic_description", "").upper()
        bio_keywords = ["PHARMA", "BIO", "DRUG", "MEDICAL", "SURGICAL", "LIFE", "HEALTH", "THERAP"]
        is_bio = any(k in industry for k in bio_keywords) or any(k in name.upper() for k in bio_keywords)
        return {"name": name, "industry": industry if industry else "General", "is_bio": is_bio}
    except:
        return {"name": ticker, "industry": "Unknown", "is_bio": False}

@st.cache_data
def get_earnings_info(ticker):
    earnings_date = None
    source = ""
    try:
        stock = yf.Ticker(ticker)
        try:
            cal = stock.calendar
            if cal and isinstance(cal, dict) and 'Earnings Date' in cal:
                earnings_date = cal['Earnings Date'][0]
        except: pass

        if not earnings_date:
            try:
                today_ts = datetime.now()
                df = stock.get_earnings_dates(limit=8)
                future = df[df.index > today_ts].sort_index()
                if not future.empty: earnings_date = future.index[0]
            except: pass
        if earnings_date: source = "Yahoo"
    except: pass

    if not earnings_date:
        try:
            url = "https://api.perplexity.ai/chat/completions"
            headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
            prompt = f"Find the next confirmed earnings release date for {ticker}. Output ONLY the date in YYYY-MM-DD format."
            payload = {"model": "sonar", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
            res = requests.post(url, json=payload, headers=headers, timeout=5)
            if res.status_code == 200:
                match = re.search(r'\d{4}-\d{2}-\d{2}', res.json()["choices"][0]["message"]["content"])
                if match:
                    earnings_date = datetime.strptime(match.group(0), "%Y-%m-%d").date()
                    source = "AI Search"
        except: pass

    if earnings_date:
        if isinstance(earnings_date, datetime): e_date = earnings_date.date()
        else: e_date = earnings_date
        days_left = (e_date - datetime.now().date()).days
        d_str = "D-Day" if days_left == 0 else f"D-{days_left}" if days_left > 0 else "완료"
        return {"date": e_date.strftime("%Y-%m-%d"), "d_day": d_str, "days_left": days_left, "source": source}
    return {"date": "미정", "d_day": "-", "days_left": 999, "source": "-"}

def get_clean_name(name):
    name = re.sub(r'[,.]', '', name)
    remove = ['Inc', 'Corp', 'Corporation', 'Ltd', 'PLC', 'Group', 'Holdings', 'Therapeutics', 'Pharma']
    for word in remove: name = re.sub(r'\b' + word + r'\b', '', name, flags=re.IGNORECASE)
    return name.strip()

def get_fda_data(company_name):
    """FDA 데이터 조회 (영어 원본 반환)"""
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
                    summary.append(f"• {r.get('report_date','-')} ({r.get('status','-')})\n  └ {r.get('reason_for_recall','')[:150]}...")
                return "\n".join(summary)
            return "✅ 최근 리콜/제재 이력 없음"
        return "ℹ️ FDA 데이터 없음"
    except: return "❌ FDA 서버 연결 실패"

def translate_to_korean(text):
    """Gemini를 이용한 한글 번역 함수"""
    if "없음" in text or "실패" in text: return text
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"다음 FDA 리콜 내역을 한국어로 자연스럽게 번역해줘. 의학 용어는 이해하기 쉽게 풀어서 써줘:\n\n{text}"
        response = model.generate_content(prompt)
        return response.text
    except: return text

def analyze_with_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.generate_content(prompt).text
    except: return "Gemini 분석 실패"

def run_hybrid_analysis(mode, system_data, fda_data, earnings_data):
    e_warn = f"\n🚨 [주의] 실적 발표 {earnings_data['d_day']} 남음!" if earnings_data['days_left'] <= 7 and earnings_data['days_left'] >= 0 else ""
    context = f"[FDA/임상 데이터]\n{fda_data}\n" if mode == "BIO" else ""
    role = "바이오 전문 펀드매니저" if mode == "BIO" else "기술주 애널리스트"

    prompt = f"""당신은 {role}입니다.

[분석 데이터]
{system_data}
{context}
[실적 일정]
다음 발표: {earnings_data['date']} ({earnings_data['d_day']}) {e_warn}

[지시사항]
1. ⚠️ 실시간 웹 검색으로 최근 24시간 내 뉴스를 확인하세요.
2. 🚫 면책 조항 금지. 분석 결과만 직설적으로 전달하세요.
3. FDA 리콜이 있다면 그 심각성을 평가하세요.

[출력 양식]
## 📰 뉴스/팩트체크
(최신 이슈 요약)

## ⚠️ 핵심 리스크
(악재, FDA, 실적 변동성 등)

## 🎯 최종 판단
(매수🟢 / 관망🟡 / 매도🔴) - (한 문장 이유)
"""
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "sonar",
        "messages": [{"role": "system", "content": "You are a helpful financial assistant."}, {"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    try: return requests.post(url, json=payload, headers=headers).json()["choices"][0]["message"]["content"]
    except Exception as e: return f"AI 분석 오류: {e}"

def extract_signal(text):
    text = text.lower()
    if "🟢" in text or "매수" in text: return "매수 기회", "#d4edda", "#155724"
    elif "🔴" in text or "매도" in text: return "위험/매도", "#f8d7da", "#721c24"
    else: return "관망 필요", "#fff3cd", "#856404"

# ==========================================
# [4] 메인 애플리케이션 로직
# ==========================================
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "analysis_data" not in st.session_state: st.session_state.analysis_data = None

st.title("📡 미국 주식 세력 탐지기")
st.caption("Auto-Mode + Earnings + Translated Data 🇰🇷")

col_input, col_btn = st.columns([2, 1])
ticker = col_input.text_input("티커 입력", value="IONQ", label_visibility="collapsed").upper().strip()
run_btn = col_btn.button("분석 실행 🚀", type="primary", use_container_width=True)

if run_btn:
    with st.spinner(f"[{ticker}] 데이터 채굴 및 AI 번역 중..."):
        try:
            client = RESTClient(API_KEY)
            info = get_ticker_info(ticker, client)
            earnings = get_earnings_info(ticker)
            mode = "BIO" if info['is_bio'] else "GENERAL"
            company_name = info['name']

            end_dt = datetime.now(pytz.timezone("America/New_York"))
            start_dt = end_dt - timedelta(days=14)
            aggs = list(client.list_aggs(ticker, 1, "minute", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), limit=50000))

            if not aggs:
                st.error(f"❌ '{ticker}' 데이터를 찾을 수 없습니다.")
            else:
                current_price = aggs[-1].close
                total_vol = sum(a.volume for a in aggs)
                total_pv = sum(((a.high+a.low+a.close)/3)*a.volume for a in aggs)
                vwap = total_pv/total_vol if total_vol else 0
                price_vol = defaultdict(int)
                for a in aggs: price_vol[round(a.close, 1)] += a.volume
                support = max(price_vol, key=price_vol.get)
                diff = ((current_price - vwap)/vwap)*100

                # FDA 데이터 수집 및 번역
                fda_info_eng = get_fda_data(company_name) if mode == "BIO" else "N/A"
                fda_info_kr = translate_to_korean(fda_info_eng) if mode == "BIO" and "없음" not in fda_info_eng else fda_info_eng

                st.session_state.analysis_data = {
                    "ticker": ticker, "name": company_name, "price": current_price, "mode": mode
                }

                # UI 표시
                badge_bg = "#e6fffa" if mode == "BIO" else "#e6f7ff"
                earnings_html = f"<span class='d-day-badge' style='margin-left:5px;'>🚨 실적 {earnings['d_day']}</span>" if earnings['days_left'] <= 7 and earnings['days_left'] >= 0 else ""
                
                st.markdown(f"""
                <div style='text-align:center; margin-bottom:15px;'>
                    <span style='background-color:{badge_bg}; padding:5px 10px; border-radius:5px; font-weight:bold; color:#444;'>{mode} MODE</span>
                    {earnings_html}
                </div>
                """, unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                c1.metric("현재가", f"${current_price}")
                c2.metric("세력평단", f"${vwap:.2f}", f"{diff:.1f}%")
                c3, c4 = st.columns(2)
                c3.metric("지지선", f"${support}")
                c4.metric("실적발표", f"{earnings['d_day']}", f"{earnings['date']}")

                # AI 분석
                sys_data = f"종목: {ticker}, 가격: {current_price}, VWAP: {vwap:.2f}, 지지선: {support}"
                gemini_res = analyze_with_gemini(f"기술적 분석 요약:\n{sys_data}")
                sys_data_full = f"{sys_data}\n[Gemini 의견]: {gemini_res}"
                ai_report = run_hybrid_analysis(mode, sys_data_full, fda_info_eng, earnings) # 분석엔 영어 데이터 사용 (정확도)

                sig_text, bg, txt = extract_signal(ai_report)
                st.markdown(f"""
                <div style="background-color:{bg}; padding:15px; border-radius:12px; text-align:center; margin:20px 0; border:1px solid {txt}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h2 style="color:{txt}; margin:0; font-size:1.6rem;">{sig_text}</h2>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("📊 AI 상세 분석 리포트", expanded=False):
                    st.markdown(ai_report)
                
                if mode == "BIO":
                    with st.expander("💊 FDA 리콜/제재 데이터 (한글 번역됨)", expanded=False):
                        st.markdown(fda_info_kr) # 번역된 한글 데이터 표시

                st.session_state.chat_history.append({"role": "assistant", "content": f"**[{ticker}] 분석결과**\n{sig_text}\n\n{ai_report}"})

        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

# ==========================================
# [5] 채팅 섹션
# ==========================================
st.divider()
st.subheader("💬 AI 투자 자문")

msgs = st.session_state.chat_history[-2:] if len(st.session_state.chat_history) > 2 else st.session_state.chat_history
for msg in msgs:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if q := st.chat_input("질문 입력"):
    st.session_state.chat_history.append({"role": "user", "content": q})
    with st.chat_message("user"): st.write(q)
    
    with st.chat_message("assistant"):
        with st.spinner("실시간 검색 중..."):
            ctx = f"[종목:{st.session_state.analysis_data['ticker']}]" if st.session_state.analysis_data else ""
            prompt = f"데이터: {ctx}\n질문: {q}\n지시: 최신뉴스기반, 면책조항금지, 짧게답변."
            h = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
            d = {"model": "sonar", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
            try:
                res = requests.post("https://api.perplexity.ai/chat/completions", json=d, headers=h).json()
                ans = res["choices"][0]["message"]["content"]
                st.markdown(ans)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
            except: st.error("응답 실패")
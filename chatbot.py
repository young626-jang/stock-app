import streamlit as st
from polygon import RESTClient
from datetime import datetime, timedelta
from collections import defaultdict
import google.generativeai as genai
import pytz
import requests
import re
import concurrent.futures

# ==========================================
# [1] 모바일 최적화 설정 (Wide 모드 + CSS)
# ==========================================
st.set_page_config(
    page_title="세력 탐지기 Pro",
    page_icon="🧬",
    layout="wide",  # 모바일 좌우 여백 제거
    initial_sidebar_state="collapsed"
)

# 모바일용 커스텀 CSS 주입
st.markdown("""
    <style>
    /* 상단 헤더 숨기기 및 여백 최소화 */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    /* 버튼 모바일 최적화 */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        font-weight: bold;
    }
    /* 메트릭 박스 디자인 */
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# [보안 설정] API 키 로드
# ==========================================
try:
    API_KEY = st.secrets["POLYGON_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
    FDA_API_KEY = st.secrets["FDA_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("🚨 API 키 설정 필요")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# [함수 정의] (기존 로직 유지)
# ==========================================
@st.cache_data
def get_available_gemini_model():
    return "gemini-1.5-flash"

def get_est_date():
    return datetime.now(pytz.timezone("America/New_York"))

def clean_company_name(name):
    name = re.sub(r'[,.]', '', name)
    remove_words = ['Inc', 'Corp', 'Corporation', 'Ltd', 'PLC', 'Group', 'Holdings']
    for word in remove_words:
        name = re.sub(r'\b' + word + r'\b', '', name, flags=re.IGNORECASE)
    return name.strip()

def get_company_name(ticker, client):
    try: return client.get_ticker_details(ticker).name
    except: return ticker

@st.cache_data(ttl=3600)  # 1시간 캐시
def get_fda_enforcements(company_name):
    clean_name = clean_company_name(company_name)
    search_query = clean_name.replace(" ", "+")
    url = f"https://api.fda.gov/drug/enforcement.json?api_key={FDA_API_KEY}&search=openfda.manufacturer_name:{search_query}&limit=3&sort=report_date:desc"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        results = response.json().get('results', [])
        if results:
            summary = []
            for res in results:
                date = res.get('report_date', '-')
                status = res.get('status', '-')
                reason = res.get('reason_for_recall', '')[:60]
                summary.append(f"• {date} ({status})\n  └ {reason}...")
            return "\n".join(summary)
        return "✅ 최근 리콜 이력 없음"
    except requests.Timeout:
        return "⏱️ FDA 타임아웃 (네트워크 느림)"
    except requests.ConnectionError:
        return "🔌 FDA 연결 실패 (인터넷 확인)"
    except Exception as e:
        return f"❌ FDA 오류: {str(e)[:30]}"

def analyze_with_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.generate_content(prompt).text
    except: return "분석 불가"

def verify_with_perplexity(gemini_analysis, system_data, fda_data):
    prompt = f"""[데이터]
{system_data}
[FDA]
{fda_data}
[Gemini]
{gemini_analysis}

[지시]
1. FDA 리콜 내역이 악재인지 확인.
2. 24시간 내 최신 뉴스 검색.
3. 면책조항 절대 금지.
4. 아래 양식으로 답변.

[양식]
## 💊 FDA/임상
(내용)
## 📰 뉴스 팩트체크
(내용)
## 🎯 결론
(매수🟢/관망🟡/매도🔴) - (한줄 이유)"""

    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "sonar",
        "messages": [{"role": "system", "content": "핵심만 요약하는 금융 전문가."}, {"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.Timeout:
        return "⏱️ Perplexity 타임아웃 - 나중에 다시 시도하세요"
    except requests.ConnectionError:
        return "🔌 Perplexity 연결 실패 - 인터넷 확인"
    except Exception as e:
        return f"❌ 분석 실패: {str(e)[:40]}"

def extract_signal(text):
    text = text.lower()
    if "🟢" in text or "매수" in text: return "매수 기회", "#d4edda", "#155724" # 배경, 글자색
    elif "🔴" in text or "매도" in text: return "위험/매도", "#f8d7da", "#721c24"
    else: return "관망 필요", "#fff3cd", "#856404"

# ==========================================
# [메인 UI] 모바일 레이아웃
# ==========================================
if "analysis_data" not in st.session_state: st.session_state.analysis_data = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

st.title("🧬 미국 주식 세력 탐지기")
st.caption("Chart + FDA + News (Mobile Ver.)")

# 입력창과 버튼
col_input, col_btn = st.columns([2, 1])
with col_input:
    ticker = st.text_input("티커", value="IONQ", label_visibility="collapsed", placeholder="티커 입력").upper().strip()
with col_btn:
    # use_container_width=True가 모바일 핵심
    run_btn = st.button("분석 🚀", type="primary", use_container_width=True) 

if run_btn:
    with st.spinner("AI가 분석 중..."):
        try:
            client = RESTClient(API_KEY)
            company_name = get_company_name(ticker, client)
            
            end_dt = get_est_date()
            start_dt = end_dt - timedelta(days=14)
            aggs = list(client.list_aggs(ticker, 1, "minute", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), limit=50000))

            if not aggs:
                st.error("데이터 없음")
            else:
                current_price = aggs[-1].close
                total_vol = sum(a.volume for a in aggs)
                total_pv = sum(((a.high+a.low+a.close)/3)*a.volume for a in aggs)
                vwap = total_pv/total_vol if total_vol else 0
                price_vol = defaultdict(int)
                for a in aggs: price_vol[round(a.close, 1)] += a.volume
                support = max(price_vol, key=price_vol.get)
                diff = ((current_price - vwap)/vwap)*100

                st.session_state.analysis_data = {"ticker": ticker, "name": company_name, "price": current_price, "vwap": vwap}

                # 모바일용 메트릭 배치 (2열 + 1열)
                m1, m2 = st.columns(2)
                m1.metric("현재가", f"${current_price}")
                m2.metric("세력평단", f"${vwap:.2f}", f"{diff:.1f}%")
                st.metric("강력 지지선", f"${support}") # 지지선은 중요하니 크게

                sys_data = f"종목: {ticker}, 가격: {current_price}, VWAP: {vwap:.2f}, 지지선: {support}"
                gemini_prompt = f"기술적 분석 요약해줘.\n{sys_data}"

                # 병렬 처리: FDA와 Gemini를 동시에 실행 (응답속도 33% 단축)
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    fda_future = executor.submit(get_fda_enforcements, company_name)
                    gemini_res = analyze_with_gemini(gemini_prompt)  # 이 시간에 FDA도 동시 실행
                    fda_info = fda_future.result()

                pplx_res = verify_with_perplexity(gemini_res, sys_data, fda_info)
                
                # 최종 신호 카드 (모바일 가독성 최적화)
                sig_text, bg_color, text_color = extract_signal(pplx_res)
                st.markdown(f"""
                <div style="background-color:{bg_color}; padding:15px; border-radius:12px; text-align:center; margin-bottom:15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    <h3 style="color:{text_color}; margin:0; font-size:1.5rem;">{sig_text}</h3>
                </div>
                """, unsafe_allow_html=True)

                # 아코디언 (기본 닫음으로 스크롤 절약)
                with st.expander("📊 상세 분석 결과 보기", expanded=False):
                    st.markdown(pplx_res)
                
                with st.expander("💊 FDA 리콜 내역", expanded=False):
                    st.text(fda_info) # text로 해서 가독성 확보

                st.session_state.chat_history.append({"role": "assistant", "content": f"[{ticker} 결과] {sig_text}\n{pplx_res}"})

        except Exception as e:
            st.error(f"오류: {e}")

# ==========================================
# [채팅] 하단 고정 느낌
# ==========================================
st.divider()
st.subheader("💬 AI 질문")

# 최신 메시지 2개만 보여주기 (모바일 공간 절약)
recent_msgs = st.session_state.chat_history[-2:] if len(st.session_state.chat_history) > 2 else st.session_state.chat_history
if len(st.session_state.chat_history) > 2:
    st.caption(f"이전 대화 {len(st.session_state.chat_history)-2}개 숨김")

for msg in recent_msgs:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if question := st.chat_input("질문 입력 (예: 악재 있어?)"):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("검색 중..."):
            context = ""
            if st.session_state.analysis_data:
                analysis_data = st.session_state.analysis_data
                context = f"[종목:{analysis_data['ticker']}]"

            prompt = f"데이터:{context}\n질문:{question}\n지시:최신뉴스기반,면책조항금지,짧게답변."
            headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
            chat_payload = {
                "model": "sonar",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
            try:
                response = requests.post("https://api.perplexity.ai/chat/completions", json=chat_payload, headers=headers, timeout=30).json()
                answer = response["choices"][0]["message"]["content"]
                st.write(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except requests.Timeout:
                st.error("⏱️ 요청 타임아웃 (다시 시도하세요)")
            except requests.ConnectionError:
                st.error("🔌 네트워크 연결 실패")
            except Exception as e:
                st.error(f"오류: {str(e)[:50]}")

import streamlit as st
from polygon import RESTClient
from datetime import datetime, timedelta
from collections import defaultdict
import google.generativeai as genai
import pytz
import requests
import re

# ==========================================
# [기본 설정] 페이지 UI 구성
# ==========================================
st.set_page_config(
    page_title="미국 주식 세력 탐지기 (w. FDA)",
    page_icon="🧬",
    layout="centered"
)

# ==========================================
# [보안 설정] API 키 로드
# ==========================================
try:
    API_KEY = st.secrets["POLYGON_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
    FDA_API_KEY = st.secrets["FDA_API_KEY"] # FDA 키 추가됨
except (FileNotFoundError, KeyError):
    st.error("🚨 API 키 설정 오류!")
    st.write("secrets.toml 파일에 FDA_API_KEY를 추가해주세요.")
    st.stop()

# Gemini 설정
genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# [유틸리티 함수]
# ==========================================
@st.cache_data
def get_available_gemini_model():
    """Gemini 모델 선택"""
    try:
        models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
        priority = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        for p in priority:
            if any(p in m for m in models): return p
        return "gemini-1.5-flash"
    except:
        return "gemini-1.5-flash"

def get_est_date():
    return datetime.now(pytz.timezone("America/New_York"))

def clean_company_name(name):
    """FDA 검색률을 높이기 위해 회사 이름 단순화 (Inc, Corp 제거)"""
    name = re.sub(r'[,.]', '', name) # 특수문자 제거
    remove_words = ['Inc', 'Corp', 'Corporation', 'Ltd', 'PLC', 'Group', 'Holdings']
    for word in remove_words:
        name = re.sub(r'\b' + word + r'\b', '', name, flags=re.IGNORECASE)
    return name.strip()

# ==========================================
# [데이터 수집 함수] Polygon, FDA
# ==========================================
def get_company_name(ticker, client):
    """Polygon에서 티커로 회사 풀네임 조회"""
    try:
        details = client.get_ticker_details(ticker)
        return details.name
    except:
        return ticker # 실패 시 티커 그대로 반환

def get_fda_enforcements(company_name):
    """FDA API: 최근 리콜/제재(Enforcement) 이력 조회"""
    clean_name = clean_company_name(company_name)
    # 검색어 공백을 +로 치환
    search_query = clean_name.replace(" ", "+")
    
    # FDA Enforcement API (리콜 정보)
    url = f"https://api.fda.gov/drug/enforcement.json?api_key={FDA_API_KEY}&search=openfda.manufacturer_name:{search_query}&limit=3&sort=report_date:desc"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            if results:
                summary = []
                for res in results:
                    date = res.get('report_date', '날짜미상')
                    reason = res.get('reason_for_recall', '사유 없음')
                    status = res.get('status', '상태 미상')
                    summary.append(f"- [{date}] {status}: {reason[:100]}...")
                return "\n".join(summary)
            else:
                return "최근 리콜/제재 이력 없음 (양호)"
        else:
            return "FDA 데이터 없음 (검색결과 없음)"
    except Exception as e:
        return f"FDA 조회 실패: {e}"

# ==========================================
# [AI 분석 함수]
# ==========================================
def analyze_with_gemini(prompt):
    try:
        model = genai.GenerativeModel(get_available_gemini_model())
        return model.generate_content(prompt).text
    except:
        return "Gemini 분석 불가"

def verify_with_perplexity(gemini_analysis, system_data, fda_data):
    """Perplexity: 차트 + 뉴스 + FDA 데이터 통합 분석"""

    prompt = f"""당신은 바이오 및 금융 리스크 관리 전문가입니다.

[분석 데이터]
{system_data}

[💊 FDA 공식 데이터 (리콜/제재)]
{fda_data}

[📈 기술적 분석(Gemini)]
{gemini_analysis}

---
[필수 지시사항]
1. **FDA 데이터 분석**: 위 FDA 데이터가 주가에 악재인지 호재인지 판단하세요. (데이터가 없으면 '특이사항 없음'으로 간주)
2. **실시간 검색**: 최근 24시간 내 뉴스를 검색하여 임상 결과, FDA 승인, 경쟁사 이슈 등을 확인하세요.
3. **면책 조항 금지**: 쫄지 말고 분석 결과만 직설적으로 말하세요.

[출력 양식]
1. **💊 FDA/임상 리스크**: (FDA 데이터 및 임상 관련 뉴스 분석)
2. **📰 실시간 뉴스 팩트체크**: (24시간 내 주요 이슈)
3. **🎯 최종 판결**: (매수🟢 / 관망🟡 / 매도🔴)
   - 이유: (한 문장 요약)

결과는 한국어로 작성하세요."""

    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "당신은 FDA 데이터와 금융 정보를 결합하여 분석하는 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    
    try:
        res = requests.post(url, json=payload, headers=headers).json()
        return res["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Perplexity 분석 실패: {e}"

def extract_signal(text):
    text = text.lower()
    if "🟢" in text or "매수" in text: return "🟢 매수 기회", "green"
    elif "🔴" in text or "매도" in text: return "🔴 위험/매도", "red"
    else: return "🟡 관망 필요", "orange"

# ==========================================
# [메인 로직]
# ==========================================
if "analysis_data" not in st.session_state: st.session_state.analysis_data = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

st.title("🧬 미국 주식 세력 탐지기 (Pro)")
st.caption("Polygon (Chart) + FDA (Bio Data) + Perplexity (News)")

ticker = st.text_input("종목 코드 (예: PFE, LLY, NVDA)", value="PFE").upper().strip()

if st.button("🧬 FDA 데이터 포함 정밀 분석", type="primary"):
    with st.spinner(f"[{ticker}] 차트, 뉴스, 그리고 FDA 서버를 터는 중..."):
        try:
            # 1. Polygon 연결 및 데이터 수집
            client = RESTClient(API_KEY)
            company_name = get_company_name(ticker, client) # 회사 이름 조회
            
            end_dt = get_est_date()
            start_dt = end_dt - timedelta(days=14)
            aggs = list(client.list_aggs(ticker, 1, "minute", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), limit=50000))

            if not aggs:
                st.error("데이터 없음. 티커 확인.")
            else:
                # 2. 계산
                current_price = aggs[-1].close
                total_vol = sum(a.volume for a in aggs)
                total_pv = sum(((a.high+a.low+a.close)/3)*a.volume for a in aggs)
                vwap = total_pv/total_vol if total_vol else 0
                price_vol = defaultdict(int)
                for a in aggs: price_vol[round(a.close, 1)] += a.volume
                support = max(price_vol, key=price_vol.get)
                diff = ((current_price - vwap)/vwap)*100

                # 3. FDA 데이터 조회
                st.toast(f"FDA에서 '{company_name}' 조회 중...", icon="💊")
                fda_info = get_fda_enforcements(company_name)

                # 4. 저장 및 표시
                data = {"ticker": ticker, "name": company_name, "price": current_price, "vwap": vwap}
                st.session_state.analysis_data = data

                c1, c2, c3 = st.columns(3)
                c1.metric("종목명", company_name)
                c2.metric("현재가", f"${current_price}")
                c3.metric("세력평단", f"${vwap:.2f}", f"{diff:.2f}%")
                
                # 5. AI 분석
                sys_data = f"종목: {ticker}({company_name}), 가격: ${current_price}, VWAP: ${vwap:.2f}, 지지선: ${support}"
                gemini_res = analyze_with_gemini(f"이 주식 기술적 분석해줘.\n{sys_data}")
                
                pplx_res = verify_with_perplexity(gemini_res, sys_data, fda_info)
                
                # 결과 출력
                with st.expander("💊 FDA 리콜/제재 데이터 (Raw Data)", expanded=True):
                    st.info(fda_info)
                
                with st.expander("📊 Gemini 기술적 분석", expanded=False):
                    st.write(gemini_res)
                    
                st.subheader("🤖 AI 최종 분석 결과")
                st.write(pplx_res)

                # 신호 박스
                sig, col = extract_signal(pplx_res)
                st.markdown(f"""
                <div style="padding:15px; border:2px solid {col}; border-radius:10px; text-align:center; background-color:{'#f0fff4' if col=='green' else '#fff5f5' if col=='red' else '#fffaf0'}">
                    <h2 style="color:{col}; margin:0;">{sig}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.chat_history.append({"role": "assistant", "content": f"[{ticker} 분석]\n{pplx_res}"})

        except Exception as e:
            st.error(f"오류: {e}")

# ==========================================
# [채팅]
# ==========================================
st.divider()
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if q := st.chat_input("질문하세요 (예: FDA 승인 언제야?)"):
    st.session_state.chat_history.append({"role": "user", "content": q})
    with st.chat_message("user"): st.write(q)
    
    with st.chat_message("assistant"):
        with st.spinner("FDA 데이터베이스 및 뉴스 검색 중..."):
            ctx = ""
            if st.session_state.analysis_data:
                d = st.session_state.analysis_data
                ctx = f"[종목: {d['ticker']}, 회사: {d['name']}]"
            
            p = f"데이터: {ctx}\n질문: {q}\n지시: FDA 이슈와 최신 뉴스를 포함해 답변. 면책조항 금지."
            h = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
            d = {"model": "sonar", "messages": [{"role": "user", "content": p}], "temperature": 0.2}
            r = requests.post("https://api.perplexity.ai/chat/completions", json=d, headers=h).json()
            ans = r["choices"][0]["message"]["content"]
            st.write(ans)
            st.session_state.chat_history.append({"role": "assistant", "content": ans})

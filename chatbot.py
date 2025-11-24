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
# [SIC 코드 기반 섹터 분류]
# ==========================================
# FDA 규제 대상 SIC 코드 (의약품, 의료기기, 바이오)
BIO_SIC_CODES = {
    2834,  # Pharmaceutical Preparations
    2835,  # In Vitro and In Vivo Diagnostic Substances
    2836,  # Biological Products, Except Diagnostic Substances
    3842,  # Orthopedic, Prosthetic, and Surgical Appliances
    3845,  # Electromedical and Electrotherapeutic Apparatus
    3851,  # Ophthalmic Goods
}

# ==========================================
# [함수 정의] (기존 로직 + SIC 개선)
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

@st.cache_data(ttl=3600)
def get_ticker_info_detailed(ticker, _client):
    """
    SIC 코드 기반 정확한 섹터 분류
    BIO (FDA 규제) vs GENERAL (비규제)
    """
    try:
        details = _client.get_ticker_details(ticker)
        name = details.name

        # SIC 코드 추출 (정수형)
        sic_code = None
        if hasattr(details, 'sic'):
            try:
                sic_code = int(details.sic) if details.sic else None
            except (ValueError, TypeError):
                sic_code = None

        # SIC 코드 기반 1차 판단
        if sic_code and sic_code in BIO_SIC_CODES:
            is_bio = True
            reason = f"SIC {sic_code}"
        # 백업: 회사명 기반 2차 판단
        else:
            bio_keywords = ["PHARMA", "BIO", "DRUG", "MEDICAL", "SURGICAL", "THERAP", "BIOTECH"]
            is_bio = any(k in name.upper() for k in bio_keywords)
            reason = "Name Pattern" if is_bio else "Non-Bio"

        return {
            "name": name,
            "sic_code": sic_code,
            "is_bio": is_bio,
            "reason": reason
        }
    except Exception:
        return {
            "name": ticker,
            "sic_code": None,
            "is_bio": False,
            "reason": "Error"
        }

def get_company_name(ticker, client):
    try:
        return client.get_ticker_details(ticker).name
    except:
        return ticker

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
    with st.spinner("섹터 분석 및 데이터 채굴 중..."):
        try:
            client = RESTClient(API_KEY)

            # 1️⃣ SIC 코드 기반 정확한 섹터 분류
            ticker_info = get_ticker_info_detailed(ticker, client)
            company_name = ticker_info["name"]
            sic_code = ticker_info["sic_code"]
            is_bio = ticker_info["is_bio"]
            classification_reason = ticker_info["reason"]

            # 사용자에게 현재 모드 알려주기
            if is_bio:
                st.toast(f"🧬 BIO 섹터 감지 ({classification_reason})! FDA 데이터베이스 연결...", icon="💊")
            else:
                st.toast(f"💻 GENERAL 섹터 ({classification_reason}) - 뉴스 & 실적 분석...", icon="🏢")

            # 2️⃣ 차트 데이터 수집
            end_dt = get_est_date()
            start_dt = end_dt - timedelta(days=14)
            aggs = list(client.list_aggs(ticker, 1, "minute", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), limit=50000))

            if not aggs:
                st.error(f"❌ '{ticker}' 데이터를 찾을 수 없습니다.")
            else:
                # 3️⃣ 기술적 지표 계산
                current_price = aggs[-1].close
                total_vol = sum(a.volume for a in aggs)
                total_pv = sum(((a.high+a.low+a.close)/3)*a.volume for a in aggs)
                vwap = total_pv/total_vol if total_vol else 0
                price_vol = defaultdict(int)
                for a in aggs: price_vol[round(a.close, 1)] += a.volume
                support = max(price_vol, key=price_vol.get)
                diff = ((current_price - vwap)/vwap)*100

                # 4️⃣ 데이터 세션 저장
                st.session_state.analysis_data = {
                    "ticker": ticker,
                    "name": company_name,
                    "price": current_price,
                    "is_bio": is_bio,
                    "sic_code": sic_code
                }

                # 5️⃣ 화면 표시 - 섹터 배지
                badge_color = "#e6fffa" if is_bio else "#e6f7ff"
                badge_text = "🧬 BIO/PHARMA" if is_bio else "💻 TECH/GENERAL"
                sic_display = f" (SIC {sic_code})" if sic_code else " (No SIC)"
                st.markdown(f"""
                <div style='text-align:center; background-color:{badge_color}; padding:8px; border-radius:8px; margin-bottom:15px; font-size:0.9rem; color:#333; font-weight:bold;'>
                    {badge_text}{sic_display}
                </div>
                """, unsafe_allow_html=True)

                # 모바일용 메트릭 배치
                m1, m2 = st.columns(2)
                m1.metric("현재가", f"${current_price}")
                m2.metric("세력평단", f"${vwap:.2f}", f"{diff:.1f}%")
                st.metric("강력 지지선", f"${support}")

                # 6️⃣ AI 분석 (모드별 조건부 프롬프트)
                sys_data = f"종목: {ticker}({company_name}), SIC: {sic_code if sic_code else 'N/A'}, 가격: {current_price}, VWAP: {vwap:.2f}, 지지선: {support}"

                # BIO 모드: FDA 필수, GENERAL 모드: FDA 스킵
                if is_bio:
                    # 병렬 처리: FDA와 Gemini 동시 실행
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                        fda_future = executor.submit(get_fda_enforcements, company_name)
                        gemini_res = analyze_with_gemini(f"생명공학 회사의 기술적 분석\n{sys_data}")
                        fda_info = fda_future.result()

                    # BIO용 프롬프트
                    pplx_prompt = f"""[BIO 섹터 분석]
[종목 데이터]
{sys_data}

[FDA 규제 현황]
{fda_info}

[기술적 분석]
{gemini_res}

[분석 지시]
1. FDA 리콜/제재가 실제 주가에 미칠 영향도 분석
2. 최근 임상 결과, PDUFA 날짜, 파이프라인 이슈 검색 (24시간)
3. 기술적 위치(VWAP, 지지선)와 결합하여 종합 판단
4. 면책조항 금지

[결과 양식]
## 💊 FDA/임상 이슈
(FDA 리콜 내역과 영향도)

## 📰 바이오 뉴스 체크
(최근 뉴스 요약)

## 🎯 결론
(매수🟢/관망🟡/매도🔴) - (한줄 이유)"""
                else:
                    # GENERAL 모드: FDA 제외
                    gemini_res = analyze_with_gemini(f"기술주/성장주 기술적 분석\n{sys_data}")
                    fda_info = "해당 없음 (Non-Bio Sector)"

                    # GENERAL용 프롬프트
                    pplx_prompt = f"""[일반/기술 섹터 분석]
[종목 데이터]
{sys_data}

[기술적 분석]
{gemini_res}

[분석 지시]
1. 최신 실적, CEO 발언, 제품 출시, 계약 공시 검색 (24시간)
2. 거시 경제 영향 및 경쟁 구도 분석
3. 기술적 위치(VWAP, 지지선)와 결합하여 판단
4. 면책조항 금지

[결과 양식]
## 🏢 펀더멘탈/뉴스 이슈
(실적, 뉴스, 공시 요약)

## ⚠️ 리스크 체크
(발견된 위험요소)

## 🎯 결론
(매수🟢/관망🟡/매도🔴) - (한줄 이유)"""

                # Perplexity 분석 (모드별 다른 프롬프트)
                url = "https://api.perplexity.ai/chat/completions"
                headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "model": "sonar",
                    "messages": [
                        {"role": "system", "content": f"당신은 {'바이오/제약 투자 전문가' if is_bio else '기술주/성장주 전문가'}입니다. 팩트 기반으로 직설적 답변. 면책조항 금지."},
                        {"role": "user", "content": pplx_prompt}
                    ],
                    "temperature": 0.2
                }

                try:
                    pplx_res = requests.post(url, json=payload, headers=headers, timeout=30).json()["choices"][0]["message"]["content"]
                except Exception as e:
                    pplx_res = f"❌ 분석 실패: {str(e)[:40]}"
                
                # 최종 신호 카드
                sig_text, bg_color, text_color = extract_signal(pplx_res)
                st.markdown(f"""
                <div style="background-color:{bg_color}; padding:15px; border-radius:12px; text-align:center; margin:15px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    <h3 style="color:{text_color}; margin:0; font-size:1.5rem;">{sig_text}</h3>
                </div>
                """, unsafe_allow_html=True)

                # 분석 결과 아코디언
                with st.expander(f"📊 {'💊 BIO' if is_bio else '🏢 GENERAL'} 상세 분석 결과", expanded=False):
                    st.markdown(pplx_res)

                # BIO 모드일 때만 FDA 아코디언 표시
                if is_bio:
                    with st.expander("💊 FDA 규제 데이터 원본", expanded=False):
                        st.text(fda_info)
                else:
                    with st.expander("ℹ️ 섹터 분류 정보", expanded=False):
                        st.info(f"**분류:** GENERAL (Non-Bio)\n**SIC 코드:** {sic_code if sic_code else 'N/A'}\n**판단 기준:** {classification_reason}")

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"[{ticker}/{('BIO' if is_bio else 'GENERAL')}] {sig_text}\n\n{pplx_res}"
                })

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

if question := st.chat_input("질문 입력 (예: 악재 있어?, 목표가는?)"):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("분석 중..."):
            context = ""
            mode_hint = ""

            if st.session_state.analysis_data:
                data = st.session_state.analysis_data
                ticker = data["ticker"]
                is_bio = data.get("is_bio", False)
                context = f"[종목:{ticker}, 모드:{'BIO' if is_bio else 'GENERAL'}]"

                # 모드별 힌트
                mode_hint = "바이오 회사의 FDA/임상 이슈를 중심으로" if is_bio else "기술주의 실적/뉴스를 중심으로"

            # 모드별 프롬프트
            chat_prompt = f"""데이터:{context}
질문:{question}

지시:
- {mode_hint} 답변
- 최신 뉴스 검색 (24시간 우선)
- 면책조항 금지
- 짧고 명확하게 (3줄 이내)"""

            headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
            chat_payload = {
                "model": "sonar",
                "messages": [{"role": "user", "content": chat_prompt}],
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

import streamlit as st
from polygon import RESTClient
from datetime import datetime, timedelta
from collections import defaultdict
import google.generativeai as genai
import pytz
import requests

# ==========================================
# [기본 설정] 페이지 UI 구성
# ==========================================
st.set_page_config(
    page_title="미국 주식 세력 탐지기 (w. Perplexity)",
    page_icon="🚀",
    layout="centered"
)

# ==========================================
# [보안 설정] API 키 로드
# ==========================================
try:
    API_KEY = st.secrets["POLYGON_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("🚨 API 키가 설정되지 않았습니다!")
    st.code("""
    # .streamlit/secrets.toml 파일에 아래 내용을 추가하세요
    POLYGON_API_KEY = "..."
    GEMINI_API_KEY = "..."
    PERPLEXITY_API_KEY = "..."
    """, language="toml")
    st.stop()

# Gemini 설정
genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# [핵심 함수] 데이터 처리 및 AI 분석
# ==========================================
@st.cache_data
def get_available_gemini_model():
    """Gemini 모델 자동 선택 (2.0 -> 1.5 Flash -> 1.5 Pro)"""
    try:
        models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
        priority = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        for p in priority:
            if any(p in m for m in models): return p
        return "gemini-1.5-flash"
    except:
        return "gemini-1.5-flash"

def get_est_date():
    """미국 동부 표준시(EST) 기준 날짜 반환"""
    return datetime.now(pytz.timezone("America/New_York"))

def analyze_with_gemini(prompt):
    """1단계: Gemini (데이터/차트 기술적 분석)"""
    try:
        model_name = get_available_gemini_model()
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini 분석 실패: {e}"

def verify_with_perplexity(gemini_analysis, system_data):
    """2단계: Perplexity (실시간 뉴스 교차 검증 및 리스크 체크)"""
    
    # 강력한 지시사항이 포함된 프롬프트
    prompt = f"""당신은 월스트리트의 냉철한 리스크 관리자입니다.

[현재 종목 데이터]
{system_data}

[기술적 분석가(Gemini)의 의견]
{gemini_analysis}

---
[필수 지시사항]
1. ⚠️ **실시간 웹 검색**을 통해 이 종목의 **최근 24시간 뉴스**를 반드시 확인하세요.
2. Gemini의 분석이 현재 시장 분위기나 최신 공시와 일치하는지 팩트 체크하세요.
3. 차트에는 보이지 않는 **돌발 악재(CEO 리스크, 실적 발표, 소송, 규제)**가 있는지 찾아내세요.

[출력 제한 - 매우 중요]
🚫 **면책 조항 금지**: "투자는 본인의 책임입니다", "이 정보는 조언이 아닙니다" 같은 상투적인 문구를 **절대 출력하지 마세요.**
🚫 분석 결과와 핵심 근거만 담백하고 직설적으로 말하세요.

[최종 산출물 양식]
1. **📰 최신 뉴스 체크**: (최근 24시간 내 주요 뉴스 3줄 요약)
2. **⚠️ 리스크 분석**: (발견된 악재나 불안 요소)
3. **🎯 최종 투자 의견**: (매수🟢 / 관망🟡 / 매도🔴 중 택1)
   - 이유: (한 문장으로 명확하게)

결과는 한국어로 작성하세요."""

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # sonar 모델: 실시간 검색 특화
    payload = {
        "model": "sonar", 
        "messages": [
            {"role": "system", "content": "당신은 최신 금융 정보를 실시간으로 검색하여 팩트 체크를 수행하는 전문가입니다."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2 # 환각 방지를 위해 낮게 설정
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Perplexity 분석 실패: {e}"

def extract_signal(text):
    """결과 텍스트에서 신호 색상 추출"""
    text = text.lower()
    if "🟢" in text or "매수" in text:
        return "🟢 매수 기회", "green"
    elif "🔴" in text or "매도" in text:
        return "🔴 위험/매도", "red"
    else:
        return "🟡 관망 필요", "orange"

# ==========================================
# [UI] 메인 화면
# ==========================================
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🚀 미국 주식 세력 탐지기")
st.markdown("##### Powered by **Gemini (Data)** + **Perplexity (Live News)**")

# 종목 입력
ticker = st.text_input("종목 코드 (예: NVDA, PLTR)", value="IONQ").upper().strip()

if st.button("🔍 세력 분석 & 뉴스 교차 검증 시작", type="primary"):
    with st.spinner(f"[{ticker}] 데이터 수집 및 AI 정밀 분석 중..."):
        try:
            # 1. Polygon 데이터 수집 (최근 14일 분봉)
            client = RESTClient(API_KEY)
            end_dt = get_est_date()
            start_dt = end_dt - timedelta(days=14)

            aggs = []
            # limit=50000은 한 번 호출 한도. 14일치 분봉은 보통 범위 내에 들어옴.
            for a in client.list_aggs(ticker, 1, "minute", start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), limit=50000):
                aggs.append(a)

            if not aggs:
                st.error(f"❌ '{ticker}' 데이터를 찾을 수 없습니다. 티커를 확인해주세요.")
            else:
                # 2. 기술적 지표 계산 (VWAP, 지지선)
                current_price = aggs[-1].close
                total_vol = sum(a.volume for a in aggs)
                total_pv = sum(((a.high+a.low+a.close)/3) * a.volume for a in aggs)
                vwap = total_pv / total_vol if total_vol else 0

                # 매물대(Volume Profile) 계산
                price_vol = defaultdict(int)
                for a in aggs:
                    price_vol[round(a.close, 1)] += a.volume
                support_price = max(price_vol, key=price_vol.get) if price_vol else 0
                diff = ((current_price - vwap) / vwap) * 100

                # 3. 데이터 세션 저장
                data = {
                    "ticker": ticker, "price": current_price, "vwap": vwap,
                    "support": support_price, "diff": diff, "vol": total_vol
                }
                st.session_state.analysis_data = data

                # 4. 결과 지표 표시
                c1, c2, c3 = st.columns(3)
                c1.metric("현재 주가", f"${current_price}")
                c2.metric("세력 평단 (VWAP)", f"${vwap:.2f}", f"{diff:.2f}%")
                c3.metric("최대 지지선", f"${support_price}")
                
                st.divider()

                # 5. Hybrid AI 분석 실행
                system_info = f"종목: {ticker}, 현재가: ${current_price}, VWAP: ${vwap:.2f}, 지지선: ${support_price}, 괴리율: {diff:.2f}%"
                
                # Gemini (기술적 분석)
                gemini_prompt = f"이 주식 데이터를 분석해줘. 세력의 평단가와 지지선을 고려할 때 기술적으로 매수 구간인지 분석해.\n{system_info}"
                gemini_res = analyze_with_gemini(gemini_prompt)
                
                # Perplexity (뉴스 검증)
                st.toast("📡 Perplexity가 실시간 뉴스를 검색하고 있습니다...", icon="🔍")
                pplx_res = verify_with_perplexity(gemini_res, system_info)
                
                # 6. 결과 출력 (아코디언 형태)
                with st.expander("📊 1단계: Gemini 기술적 분석 보고서", expanded=True):
                    st.write(gemini_res)
                
                with st.expander("🌐 2단계: Perplexity 실시간 뉴스 & 리스크 분석", expanded=True):
                    st.write(pplx_res)
                
                # 최종 신호 카드
                signal_text, color = extract_signal(pplx_res)
                st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color: {'#e6fffa' if color=='green' else '#fff5f5' if color=='red' else '#fffaf0'}; border: 1px solid {color}; text-align: center;">
                    <h3 style="color: {color}; margin:0;">{signal_text}</h3>
                </div>
                """, unsafe_allow_html=True)

                # 채팅 기록에 자동 저장
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"**[{ticker} 분석 완료]**\n\n결론: {signal_text}\n\n{pplx_res}"
                })

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")

# ==========================================
# [UI] AI 채팅 (Perplexity 기반)
# ==========================================
st.divider()
st.subheader("💬 AI 투자 자문 (실시간 검색)")

# 채팅 히스토리 렌더링
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 처리
if prompt := st.chat_input("궁금한 점을 물어보세요 (예: 오늘 왜 떨어진거야?)"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 분석된 데이터가 있으면 컨텍스트로 제공
    context_str = ""
    if st.session_state.analysis_data:
        d = st.session_state.analysis_data
        context_str = f"[분석 대상: {d['ticker']}, 가격: ${d['price']}, VWAP: ${d['vwap']:.2f}]"

    # Perplexity 채팅 요청
    with st.chat_message("assistant"):
        with st.spinner("Perplexity가 최신 정보를 검색 중입니다..."):
            
            chat_payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "system", 
                        "content": "당신은 실시간 금융 정보를 검색하여 답변하는 AI입니다. 반드시 최신 뉴스(최근 24시간)를 우선적으로 확인하고 답변하세요."
                    },
                    {
                        "role": "user", 
                        "content": f"이전 대화 맥락과 데이터를 참고하세요.\n데이터: {context_str}\n질문: {prompt}"
                    }
                ],
                "temperature": 0.3
            }
            
            try:
                headers = {"Authorization": f"Bearer {PERPLEXITY_API_KEY}", "Content-Type": "application/json"}
                res = requests.post("https://api.perplexity.ai/chat/completions", json=chat_payload, headers=headers).json()
                bot_reply = res["choices"][0]["message"]["content"]
                
                st.markdown(bot_reply)
                st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
            except Exception as e:
                st.error(f"채팅 오류: {e}")

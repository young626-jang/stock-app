import streamlit as st
from polygon import RESTClient
from datetime import datetime, timedelta
from collections import defaultdict
import google.generativeai as genai
from PIL import Image
import io
import pytz
import requests
import json

# ==========================================
# [기본 설정] 페이지 제목 및 아이콘
# ==========================================
st.set_page_config(
    page_title="미국 주식 세력 탐지기",
    page_icon="🚀",
    layout="centered"
)

# ==========================================
# [보안 설정] 환경변수(Secrets)에서 API 키 가져오기
# ==========================================
try:
    # 내 컴퓨터의 .streamlit/secrets.toml 또는 웹 서버의 Secrets에서 가져옴
    API_KEY = st.secrets["POLYGON_API_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
except (FileNotFoundError, KeyError) as e:
    st.error("🚨 API 키를 찾을 수 없습니다!")
    st.warning("`.streamlit/secrets.toml` 파일에 다음을 추가해주세요:")
    st.write("""
    - POLYGON_API_KEY
    - GEMINI_API_KEY
    - PERPLEXITY_API_KEY
    """)
    st.stop()

# Gemini API 초기화
genai.configure(api_key=GEMINI_API_KEY)

# ==========================================
# [유틸리티 함수]
# ==========================================
@st.cache_data
def get_available_model():
    """사용 가능한 Gemini 모델을 캐싱하여 조회"""
    try:
        available = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]
        model_priority = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

        for model_name in model_priority:
            if any(model_name in m for m in available):
                return model_name
        return "gemini-1.5-flash"  # 기본값
    except Exception as e:
        st.error(f"모델 조회 실패: {e}")
        return "gemini-1.5-flash"

def get_est_date():
    """미국 동부 표준시(EST) 기준 현재 시간 반환"""
    est = pytz.timezone("America/New_York")
    return datetime.now(est)

def analyze_with_gemini(prompt):
    """Gemini 1차 분석 - 빠른 기술적 분석"""
    try:
        selected_model = get_available_model()
        model = genai.GenerativeModel(selected_model)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini 분석 오류: {str(e)}"

def verify_with_gpt(gemini_analysis, system_data):
    """Perplexity 2차 검증 - Gemini 의견 비판 및 최종 신호"""
    try:
        verification_prompt = f"""당신은 냉철한 헤지펀드 포트폴리오 매니저입니다.

[분석 대상 종목 데이터]
{system_data}

[Gemini의 1차 분석 결과]
{gemini_analysis}

---

[당신의 임무]
1. **Gemini 분석 평가**: 위 분석에서 논리적 강점과 약점을 지적하세요.
2. **리스크 체크**: 숨어있는 리스크나 반박 가능한 부분을 언급하세요.
3. **최종 신호**: 아래 중 하나를 선택하여 한 문장으로 이유를 설명하세요.
   - 🟢 **매수** (강력한 매수 신호)
   - 🟡 **관망** (더 정보 필요)
   - 🔴 **매도** (매도/회피 신호)

답변은 명확하고 간결하게 한국어로 제공하세요."""

        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {"role": "system", "content": "당신은 리스크 관리 전문가이며, 객관적이고 냉철한 판단을 내립니다."},
                {"role": "user", "content": verification_prompt}
            ],
            "temperature": 0.8,
            "max_tokens": 1200
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Perplexity 검증 오류: {str(e)}"

def extract_signal(gpt_response):
    """GPT 응답에서 최종 신호 추출"""
    response_lower = gpt_response.lower()

    if "🟢" in gpt_response or "매수" in response_lower and "강" in response_lower:
        return "🟢 매수", "green"
    elif "🔴" in gpt_response or "매도" in response_lower or "회피" in response_lower:
        return "🔴 매도", "red"
    else:
        return "🟡 관망", "orange"

def hybrid_ai_analysis(user_prompt, system_data):
    """하이브리드 방식: Gemini (1차) → GPT (2차) → 최종 신호"""
    # 1단계: Gemini 1차 분석
    gemini_result = analyze_with_gemini(user_prompt)

    # 2단계: GPT 2차 검증 및 최종 신호
    gpt_result = verify_with_gpt(gemini_result, system_data)

    # 3단계: 최종 신호 추출
    signal, signal_color = extract_signal(gpt_result)

    return gemini_result, gpt_result, signal, signal_color

# ==========================================
# [메인 화면 구성]
# ==========================================
# ==========================================
# [세션 상태 초기화] - 대화 히스토리 저장
# ==========================================
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🚀 미국 주식 세력 탐지기")
st.markdown("##### 🤖 세력의 평단가(VWAP)와 지지선을 분석합니다.")

# 종목 입력창 (기본값 IONQ)
ticker = st.text_input("분석할 종목 코드 (예: NVDA, RKLB)", value="IONQ").upper().strip()

# 버튼 클릭 시 실행
if st.button("세력 의도 분석 시작 🔍", type="primary"):
    with st.spinner(f"'{ticker}' 데이터를 씹어먹는 중... 챱챱 🥣"):
        try:
            # 1. 클라이언트 연결
            client = RESTClient(API_KEY)

            # 2. 날짜 설정 (최근 14일, EST 기준)
            end_date = get_est_date().replace(hour=0, minute=0, second=0, microsecond=0)
            start_date = end_date - timedelta(days=14)
            
            str_start = start_date.strftime("%Y-%m-%d")
            str_end = end_date.strftime("%Y-%m-%d")

            # 3. 데이터 수집
            aggs = []
            for a in client.list_aggs(ticker, 1, "minute", str_start, str_end, limit=50000):
                aggs.append(a)

            # 4. 데이터 검증
            if not aggs:
                st.error(f"❌ [{ticker}] 데이터를 찾을 수 없습니다.")
                st.info("다음을 확인해주세요:")
                st.write("""
                - **티커명**: 대문자로 입력했는지 확인 (예: NVDA, TSLA)
                - **거래소**: 미국 거래소(NYSE, NASDAQ)에 상장된 종목인지 확인
                - **휴장일**: 주말이나 미국 공휴일은 데이터가 없습니다
                - **API 플랜**: Polygon API의 데이터 조회 제한을 확인하세요
                """)

            else:
                # ----------------------------------
                # 5. 핵심 분석 로직 (VWAP & 지지선)
                # ----------------------------------
                total_volume = 0
                total_pv = 0
                price_volume = defaultdict(int)
                current_price = aggs[-1].close

                for c in aggs:
                    # VWAP 계산용 (평균가 * 거래량)
                    typical_price = (c.high + c.low + c.close) / 3
                    total_pv += (typical_price * c.volume)
                    total_volume += c.volume
                    
                    # 매물대 계산 (소수점 1자리 반올림)
                    price_volume[round(c.close, 1)] += c.volume

                # 최종 계산
                vwap = total_pv / total_volume if total_volume > 0 else 0
                
                # 가장 거래량이 많았던 가격 (지지선)
                top_support = sorted(price_volume.items(), key=lambda x: x[1], reverse=True)[0][0]
                
                # 괴리율 (%)
                diff_per = ((current_price - vwap) / vwap) * 100

                # ----------------------------------
                # 6. 결과 화면 출력 (모바일 최적화)
                # ----------------------------------
                st.success("분석이 완료되었습니다!")

                # 카드 형태로 주요 지표 보여주기
                col1, col2, col3 = st.columns(3)
                col1.metric("현재 주가", f"${current_price}")
                col2.metric("세력 평단 (VWAP)", f"${vwap:.2f}", f"{diff_per:.2f}%")
                col3.metric("강력 지지선", f"${top_support}")

                st.divider() # 구분선

                # 분석 데이터 저장
                analysis_data = {
                    "ticker": ticker,
                    "current_price": current_price,
                    "vwap": vwap,
                    "top_support": top_support,
                    "diff_per": diff_per,
                    "total_volume": total_volume
                }
                st.session_state.analysis_data = analysis_data

                # 🤖 AI의 3줄 요약 판단
                st.subheader("🤖 AI의 판단")

                if current_price < top_support:
                    st.error("🚨 [위험] 지지선이 깨졌습니다!")
                    st.write(f"바닥이라고 생각했던 **${top_support}** 가격이 무너졌습니다. 지금 매수하면 물릴 확률이 높습니다.")
                elif current_price < vwap:
                    st.success("✅ [기회] 세력보다 싸게 살 기회!")
                    st.write(f"기관들의 평균 단가(**${vwap:.2f}**)보다 저렴합니다. 분할 매수하기 좋은 구간입니다.")
                else:
                    st.warning("🔥 [주의] 이미 많이 올랐습니다.")
                    st.write(f"세력들도 이미 수익 구간입니다. 추격 매수는 자제하세요.")

        except Exception as e:
            error_msg = str(e).lower()
            st.error("오류 발생!")

            if "401" in error_msg or "unauthorized" in error_msg:
                st.warning("🔑 **API 인증 오류**: Polygon API 키가 유효하지 않습니다. Secrets을 다시 확인하세요.")
            elif "429" in error_msg or "rate limit" in error_msg:
                st.warning("⏳ **API 한도 초과**: 너무 많은 요청이 발생했습니다. 잠시 후 다시 시도하세요.")
            elif "404" in error_msg or "not found" in error_msg:
                st.warning("❌ **데이터 없음**: 해당 종목의 데이터를 찾을 수 없습니다. 티커를 확인하세요.")
            else:
                st.info(f"기술 정보: {e}")

# ==========================================
# [AI 대화형 챗봇] - Gemini와의 실시간 대화
# ==========================================
st.divider()
st.subheader("💬 AI 금융 전문가와 대화하기")

if st.session_state.analysis_data:
    data = st.session_state.analysis_data
    system_prompt = f"""당신은 미국 주식 전문가입니다.

현재 분석 중인 종목: {data['ticker']}
- 현재 주가: ${data['current_price']}
- 세력 평단가(VWAP): ${data['vwap']:.2f}
- 강력 지지선: ${data['top_support']}
- 괴리율: {data['diff_per']:.2f}%
- 총 거래량: {data['total_volume']}

사용자의 질문에 대해 이 데이터를 바탕으로 자세하고 친절하게 답변해주세요.
기술적 분석, 투자 전략, 리스크 관리 등에 대해 조언할 수 있습니다.
모든 답변은 한국어로 제공하세요."""
else:
    system_prompt = "당신은 미국 주식 전문가입니다. 사용자의 투자 관련 질문에 친절하고 자세하게 답변해주세요."

# 채팅 히스토리 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
user_input = st.chat_input("투자 관련 질문을 하세요... (예: 지금 매수해도 될까요?)")

if user_input:
    # 사용자 메시지 표시
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # AI 응답 생성 (하이브리드: Gemini 1차 → GPT 2차 검증 → 최종 신호)
    with st.spinner("🤖 1단계: Gemini 분석 중..."):
        try:
            # 대화 히스토리를 프롬프트에 포함
            messages = f"{system_prompt}\n\n"
            for msg in st.session_state.chat_history[:-1]:
                role = "사용자" if msg["role"] == "user" else "전문가"
                messages += f"{role}: {msg['content']}\n\n"
            messages += f"사용자: {user_input}"

            # 시스템 데이터 준비
            if st.session_state.analysis_data:
                data = st.session_state.analysis_data
                system_data = f"""
현재 분석 중인 종목: {data['ticker']}
- 현재 주가: ${data['current_price']}
- 세력 평단가(VWAP): ${data['vwap']:.2f}
- 강력 지지선: ${data['top_support']}
- 괴리율: {data['diff_per']:.2f}%
- 총 거래량: {data['total_volume']}
"""
            else:
                system_data = ""

            # 하이브리드 분석 실행
            gemini_result, gpt_result, final_signal, signal_color = hybrid_ai_analysis(messages, system_data)

            # ==========================================
            # [결과 표시] 3단계 분석 과정
            # ==========================================

            # 1단계: Gemini 분석 결과
            with st.expander("📊 1단계: Gemini 기술적 분석 (클릭하여 확인)", expanded=False):
                st.markdown(gemini_result)

            # 2단계: Perplexity 검증 결과
            st.write("")  # 여백
            with st.expander("🔍 2단계: Perplexity 검증 및 리스크 분석 (클릭하여 확인)", expanded=False):
                st.markdown(gpt_result)

            # 3단계: 최종 신호 (강조 표시)
            st.divider()
            st.subheader("🎯 최종 투자 신호")

            # 신호를 큰 박스로 표시
            if signal_color == "green":
                st.success(f"### {final_signal}\n\n매수 기회 포착!")
            elif signal_color == "red":
                st.error(f"### {final_signal}\n\n주의 필요!")
            else:
                st.warning(f"### {final_signal}\n\n추가 정보 대기 중")

            # 최종 응답 저장
            combined_response = f"""
**1단계: Gemini 기술적 분석**
{gemini_result}

---

**2단계: Perplexity 검증 및 리스크 분석**
{gpt_result}

---

**최종 신호: {final_signal}**
"""
            st.session_state.chat_history.append({"role": "assistant", "content": combined_response})

        except Exception as e:
            st.error("AI 분석 실패!")
            error_msg = str(e).lower()
            if "api" in error_msg or "gemini" in error_msg:
                st.warning("🤖 **Gemini 오류**: Gemini API에 접속할 수 없습니다. 네트워크를 확인하세요.")
            elif "perplexity" in error_msg:
                st.warning("🟡 **Perplexity 오류**: Perplexity API에 접속할 수 없습니다. Secrets 설정을 확인하세요.")
            elif "rate limit" in error_msg or "429" in error_msg:
                st.warning("⏳ **API 한도 초과**: 너무 많은 요청이 발생했습니다. 1분 후 다시 시도하세요.")
            elif "401" in error_msg or "unauthorized" in error_msg:
                st.warning("🔑 **인증 오류**: API 키가 유효하지 않습니다. Secrets을 확인하세요.")
            else:
                st.info(f"기술 정보: {e}")
else:
    if not st.session_state.analysis_data:
        st.info("📊 먼저 종목을 분석해주세요!")

# ==========================================
# [탭 기능] - 차트 분석 & 뉴스 분석
# ==========================================
st.divider()
tab1, tab2 = st.tabs(["📊 차트 분석", "📰 뉴스 분석"])

# ==========================================
# [탭 1] 차트 업로드 분석
# ==========================================
with tab1:
    st.subheader("차트 이미지 분석")
    st.write("주식 차트 이미지를 업로드하면 AI가 기술적 분석을 수행합니다.")

    uploaded_chart = st.file_uploader(
        "차트 이미지 업로드 (PNG, JPG, GIF)",
        type=["png", "jpg", "jpeg", "gif"],
        key="chart_uploader"
    )

    if uploaded_chart is not None:
        # 이미지 표시
        image = Image.open(uploaded_chart)
        st.image(image, caption="업로드된 차트", use_column_width=True)

        # AI 분석
        if st.button("🔍 차트 분석 시작", key="analyze_chart"):
            with st.spinner("차트를 분석 중입니다..."):
                try:
                    # 사용 가능한 모델 자동 선택 (캐싱됨)
                    selected_model = get_available_model()
                    model = genai.GenerativeModel(selected_model)

                    # 이미지를 바이너리로 변환
                    image_data = uploaded_chart.getvalue()

                    # 차트 분석 프롬프트
                    analysis_prompt = """이 차트는 주식 기술적 분석 차트입니다.

다음 항목들을 분석해주세요:
1. **현재 추세** - 상승/하락/횡보 중 어느 것인가?
2. **주요 저항선/지지선** - 어디에 있는가?
3. **기술적 신호** - 매수/매도 신호가 보이는가?
4. **거래량** - 거래량 추세는 어떤가?
5. **투자 조언** - 현재 진입/청산하기 좋은 타이밍인가?

모든 분석은 한국어로 상세하게 제공해주세요."""

                    response = model.generate_content([analysis_prompt, image])
                    analysis_result = response.text

                    st.success("분석 완료!")
                    st.markdown(analysis_result)

                except Exception as e:
                    st.error("차트 분석 실패!")
                    error_msg = str(e).lower()
                    if "image" in error_msg:
                        st.warning("🖼️ **이미지 오류**: 지원하지 않는 이미지 형식입니다. PNG, JPG, GIF를 사용하세요.")
                    elif "api" in error_msg or "gemini" in error_msg:
                        st.warning("🤖 **AI 오류**: Gemini API에 접속할 수 없습니다. 네트워크를 확인하세요.")
                    else:
                        st.info(f"기술 정보: {e}")

# ==========================================
# [탭 2] 뉴스 기반 분석
# ==========================================
with tab2:
    st.subheader("주식 뉴스 분석")
    st.write("주식 관련 뉴스를 입력하면 AI가 주가에 미칠 영향을 분석합니다.")

    news_input = st.text_area(
        "분석할 뉴스를 입력하세요",
        placeholder="예: 삼성전자가 신제품 발표를 했습니다. AI 칩의 성능이 기존 제품 대비 50% 향상되었습니다...",
        height=150
    )

    news_ticker = st.text_input(
        "해당 종목 코드 (선택사항)",
        placeholder="예: NVDA, TSLA"
    )

    if st.button("📊 뉴스 영향도 분석", key="analyze_news"):
        if not news_input.strip():
            st.warning("뉴스를 입력해주세요!")
        else:
            with st.spinner("뉴스를 분석 중입니다..."):
                try:
                    # 사용 가능한 모델 자동 선택 (캐싱됨)
                    selected_model = get_available_model()
                    model = genai.GenerativeModel(selected_model)

                    # 뉴스 분석 프롬프트
                    ticker_context = f"종목: {news_ticker}\n" if news_ticker else ""
                    analysis_prompt = f"""당신은 금융 분석가입니다.

{ticker_context}다음 뉴스를 분석해주세요:

"{news_input}"

이 뉴스를 바탕으로 다음을 분석해주세요:

1. **뉴스 요약** - 이 뉴스의 핵심은 무엇인가?
2. **긍정/부정 영향** - 주가에 긍정적인지 부정적인지?
3. **영향도 수치** (1~10) - 주가에 얼마나 큰 영향을 미칠 것 같은가?
4. **영향받을 업종/종목** - 어떤 업종이나 종목이 영향받을 것인가?
5. **투자 전략** - 이를 바탕으로 어떤 투자 전략을 취해야 하는가?
6. **주의사항** - 투자할 때 주의할 점은 무엇인가?

모든 분석은 한국어로 상세하고 객관적으로 제공해주세요."""

                    response = model.generate_content(analysis_prompt)
                    news_analysis = response.text

                    st.success("분석 완료!")
                    st.markdown(news_analysis)

                except Exception as e:
                    st.error("뉴스 분석 실패!")
                    error_msg = str(e).lower()
                    if "api" in error_msg or "gemini" in error_msg:
                        st.warning("🤖 **AI 오류**: Gemini API에 접속할 수 없습니다. 네트워크를 확인하세요.")
                    elif "rate limit" in error_msg:
                        st.warning("⏳ **API 한도 초과**: 너무 많은 요청이 발생했습니다. 잠시 후 다시 시도하세요.")
                    else:
                        st.info(f"기술 정보: {e}")

import streamlit as st
from polygon import RESTClient
from datetime import datetime, timedelta
from collections import defaultdict
import google.generativeai as genai

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
except (FileNotFoundError, KeyError) as e:
    st.error("🚨 API 키를 찾을 수 없습니다!")
    st.warning("`.streamlit/secrets.toml` 파일에 POLYGON_API_KEY와 GEMINI_API_KEY를 추가해주세요.")
    st.stop()

# Gemini API 초기화
genai.configure(api_key=GEMINI_API_KEY)

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
            
            # 2. 날짜 설정 (최근 14일)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=14)
            
            str_start = start_date.strftime("%Y-%m-%d")
            str_end = end_date.strftime("%Y-%m-%d")

            # 3. 데이터 수집
            aggs = []
            for a in client.list_aggs(ticker, 1, "minute", str_start, str_end, limit=50000):
                aggs.append(a)

            # 4. 데이터 검증
            if not aggs:
                st.error(f"❌ [{ticker}] 데이터를 찾을 수 없습니다. 티커를 확인해주세요.")
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
            st.error(f"오류 발생: {e}")

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

    # AI 응답 생성
    with st.spinner("🤖 AI가 생각 중입니다..."):
        try:
            # 사용 가능한 모델 자동 선택
            available_models = [m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods]

            model_priority = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
            selected_model = "gemini-1.5-flash"  # 기본값

            for model_name in model_priority:
                if any(model_name in m for m in available_models):
                    selected_model = model_name
                    break

            model = genai.GenerativeModel(selected_model)

            # 대화 히스토리를 프롬프트에 포함
            messages = f"{system_prompt}\n\n"
            for msg in st.session_state.chat_history[:-1]:  # 마지막 사용자 메시지 제외 (이미 위에 있음)
                role = "사용자" if msg["role"] == "user" else "전문가"
                messages += f"{role}: {msg['content']}\n\n"
            messages += f"사용자: {user_input}"

            response = model.generate_content(messages)
            ai_response = response.text

            # AI 응답 저장 및 표시
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant"):
                st.markdown(ai_response)
        except Exception as e:
            st.error(f"AI 응답 생성 실패: {e}")
else:
    if not st.session_state.analysis_data:
        st.info("📊 먼저 종목을 분석해주세요!")

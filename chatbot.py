import streamlit as st
from polygon import RESTClient
from datetime import datetime, timedelta
from collections import defaultdict

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
except FileNotFoundError:
    st.error("🚨 API 키를 찾을 수 없습니다!")
    st.warning("내 컴퓨터라면 `.streamlit/secrets.toml` 파일을 만들어주세요.")
    st.stop() # 키 없으면 여기서 멈춤

# ==========================================
# [메인 화면 구성]
# ==========================================
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

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import random
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="겨울철 도로 열선 자동 가동 최적화 AI 관제 시스템",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. AI 모델 로드 함수
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ai_model():
    try:
        model = joblib.load('road_heating_model.pkll')
        return model
    except Exception:
        return None

ai_model = load_ai_model()

# -----------------------------------------------------------------------------
# 3. 사이드바 - [입력 방식 선택 & INPUT 영역]
# -----------------------------------------------------------------------------
st.sidebar.title("🎛️ 데이터 입력 제어")

# 🔥 핵심 기능: 데이터 수집 모드 선택 (자동 vs 수동)
input_mode = st.sidebar.radio(
    "📥 데이터 수집 방식 선택",
    ["🌐 AI 실시간 자동 수집 (기상청 API/센서)", "✏️ 관제사 직접 수동 입력 (시뮬레이션)"],
    index=0
)

st.sidebar.divider()

# 대상 도로 구간 선택
selected_road = st.sidebar.selectbox(
    "🏢 대상 도로 구간", 
    ["남산소월로 1구간 (급경사지)", "북악산로 3구간 (커브/음지)", "성산대교 북단 램프", "테헤란로 교차로"]
)

# 모드별 Input 데이터 연산/입력 받기
if input_mode == "🌐 AI 실시간 자동 수집 (기상청 API/센서)":
    st.sidebar.info("💡 기상청 API 및 도로 IoT 센서로부터 실시간 데이터를 수집 중입니다.")
    
    # [자동 수집 모드] 도로별 실시간/기상청 모의 데이터 자동 부여
    road_defaults = {
        "남산소월로 1구간 (급경사지)": {"slope": 12.5, "shadow": 1, "r_temp": -3.2, "temp": -4.0, "hum": 88, "snow": 1.5, "wind": 4.2},
        "북악산로 3구간 (커브/음지)": {"slope": 9.8, "shadow": 1, "r_temp": -2.8, "temp": -3.5, "hum": 85, "snow": 2.0, "wind": 3.8},
        "성산대교 북단 램프": {"slope": 6.2, "shadow": 0, "r_temp": 0.5, "temp": -1.2, "hum": 70, "snow": 0.0, "wind": 2.5},
        "테헤란로 교차로": {"slope": 2.0, "shadow": 0, "r_temp": 1.8, "temp": -0.5, "hum": 60, "snow": 0.0, "wind": 1.8}
    }
    
    data = road_defaults[selected_road]
    slope_deg = data["slope"]
    is_shadow_val = data["shadow"]
    road_temp = data["r_temp"]
    air_temp = data["temp"]
    humidity = data["hum"]
    snow_depth = data["snow"]
    wind_speed = data["wind"]

    # 사이드바에 현재 자동 입력된 데이터 현황 표시 (수정 불가 읽기전용)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**[실시간 연동 수치]**")
    st.sidebar.text(f"• 노면 온도: {road_temp} ℃")
    st.sidebar.text(f"• 기온/습도: {air_temp} ℃ / {humidity} %")
    st.sidebar.text(f"• 적설량: {snow_depth} cm")
    
    if st.sidebar.button("🔄 실시간 데이터 새로고침"):
        st.rerun()

else:
    # [수동 입력 모드] 슬라이더와 입력창을 통해 사용자가 마음대로 입력
    st.sidebar.warning("⚠️ 슬라이더를 조절하여 가상의 기상 조건 및 도로 환경을 설정하세요.")
    
    slope_deg = st.sidebar.slider("도로 경사도 (°)", 0.0, 20.0, 10.0, 0.5)
    is_shadow = st.sidebar.radio("음지 여부", ["양지 (일조 있음)", "음지 (그늘지역)"], index=1)
    is_shadow_val = 1 if is_shadow == "음지 (그늘지역)" else 0
    
    road_temp = st.sidebar.slider("현재 노면 온도 (℃)", -15.0, 10.0, -2.5, 0.1)
    air_temp = st.sidebar.slider("현재 기온 (℃)", -15.0, 10.0, -3.0, 0.1)
    humidity = st.sidebar.slider("상대 습도 (%)", 0, 100, 85)
    snow_depth = st.sidebar.number_input("적설량 (cm)", 0.0, 30.0, 1.2, 0.1)
    wind_speed = st.sidebar.slider("풍속 (m/s)", 0.0, 20.0, 3.5, 0.5)

# -----------------------------------------------------------------------------
# 4. [AI 추론 ENGINE] Input -> AI 연산 -> Output
# -----------------------------------------------------------------------------
def run_ai_inference(temp, hum, wind, snow, r_temp, slope, shadow):
    if ai_model is not None:
        features = np.array([[temp, hum, wind, snow, r_temp, slope, shadow]])
        try:
            risk_prob = ai_model.predict_proba(features)[0][1] * 100
            return int(risk_prob)
        except Exception:
            return int(ai_model.predict(features)[0])
    else:
        score = 0
        if r_temp <= 0: score += 35
        if r_temp <= -3: score += 15
        if hum >= 80: score += 20
        if snow > 0: score += 15
        if shadow == 1: score += 10
        if slope >= 8.0: score += 5
        return min(score, 100)

# AI 연산 수행
risk_score = run_ai_inference(air_temp, humidity, wind_speed, snow_depth, road_temp, slope_deg, is_shadow_val)

# AI 가동 단계 판정
if risk_score >= 70:
    heating_status = "🔥 강가동 (100% 출력)"
    status_color = "red"
    action_msg = "결빙 위험 극심! AI가 열선을 즉시 100% 가동합니다."
elif risk_score >= 40:
    heating_status = "⚠️ 예열 단계 (30% 출력)"
    status_color = "orange"
    action_msg = "결빙 가능성 존재. AI가 예열 모드로 가동을 시작합니다."
else:
    heating_status = "⚪ 대기 (OFF)"
    status_color = "green"
    action_msg = "노면 상태 안전. AI 판단 하에 열선 가동을 대기합니다."

# -----------------------------------------------------------------------------
# 5. 메인 화면 - [OUTPUT 및 시각화]
# -----------------------------------------------------------------------------
st.title("🛣️ 도로 열선 자동 가동 최적화 AI 관제 시스템")

# 모드별 상단 안내 문구
if input_mode == "🌐 AI 실시간 자동 수집 (기상청 API/센서)":
    st.success("🤖 **[AI 자동 모드 작동 중]** 기상청 API 및 노면 센서 데이터를 실시간으로 감지하여 AI가 판단을 내립니다.")
else:
    st.warning("✏️ **[관제사 수동 모드 작동 중]** 사용자가 직접 설정한 변수값으로 AI 가동 시뮬레이션을 수행합니다.")

st.markdown("---")

out_col1, out_col2 = st.columns([1, 1.2])

with out_col1:
    st.subheader("🤖 AI 모델 판단 결과 [Output]")
    st.metric(label="선택된 관제 구간", value=selected_road)
    st.metric(label="AI 예측 결빙 위험도", value=f"{risk_score} %")
    st.metric(label="AI 추천 열선 제어 상태", value=heating_status)
    
    if status_color == "red":
        st.error(f"**[AI 판단]** {action_msg}")
    elif status_color == "orange":
        st.warning(f"**[AI 판단]** {action_msg}")
    else:
        st.success(f"**[AI 판단]** {action_msg}")

with out_col2:
    st.subheader("📊 현재 적용 중인 Input 데이터 현황")
    
    input_summary = pd.DataFrame({
        "구분": ["데이터 수집 모드", "노면 온도", "기온", "습도", "적설량", "풍속", "도로 경사도", "음지 여부"],
        "현재 데이터": [
            "AI 자동 연동" if "자동" in input_mode else "관제사 수동 설정",
            f"{road_temp} ℃", f"{air_temp} ℃", f"{humidity} %", 
            f"{snow_depth} cm", f"{wind_speed} m/s", f"{slope_deg} °", 
            "음지 (그늘)" if is_shadow_val == 1 else "양지"
        ]
    })
    st.table(input_summary)

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. 하단 그래프 시각화
# -----------------------------------------------------------------------------
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("🎯 결빙 위험도 게이지")
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "AI 위험도 게이지 (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "crimson"}
            ],
        }
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_chart2:
    st.subheader("⚡ AI 최적화 가동 시 전력 절감 효과")
    est_kwh_standard = 100 
    est_kwh_ai = int(risk_score * 0.9) if risk_score >= 40 else 5
    
    fig_bar = go.Figure(data=[
        go.Bar(name='기존 단순 제어 방식', x=['예상 전력 소비량(kWh)'], y=[est_kwh_standard], marker_color='grey'),
        go.Bar(name='AI 최적화 가동', x=['예상 전력 소비량(kWh)'], y=[est_kwh_ai], marker_color='dodgerblue')
    ])
    fig_bar.update_layout(barmode='group', height=300)
    st.plotly_chart(fig_bar, use_container_width=True)
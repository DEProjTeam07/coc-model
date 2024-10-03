'''
Streamlit 웹 서비스 화면 
예측하기 버튼을 클릭하면 Flask 서버 구현을 위해 만든 predict 함수로 전달되서 
mlflow에 로드된 모델을 가지고 추론한 결과값을 Streamlit이 받아서 
사용자에게 보여준다. 
'''

import streamlit as st
import requests
import psycopg2
import os

from dotenv import load_dotenv
from PIL import Image

# .env 파일에서 환경 변수를 로드
load_dotenv()

# PostgreSQL 연결하는 함수 
def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        port=os.getenv("POSTGRES_PORT")
    )
    return conn

# DB에 좋아요/싫어요 데이터 저장 함수
def save_survey(is_good):
    conn = get_db_connection()
    cursor = conn.cursor()

    # 쿼리 실행 - created_at은 자동으로 CURRENT_TIMESTAMP로 저장됨
    cursor.execute(
        """
        INSERT INTO service_survey (is_good)
        VALUES (%s);
        """,
        (is_good,)
    )
    
    # DB 커밋 및 연결 종료
    conn.commit()
    cursor.close()
    conn.close()

# Streamlit 웹 서비스 화면 설정
st.set_page_config(page_title="타이어 상태 분류 서비스", page_icon="🚗", layout="wide")

# CSS로 페이지 상단 마진을 줄여서 헤더를 위로 올리고 좌우 구역을 아래로 내리기
st.markdown("""
    <style>
        .main-title {
            font-size:50px;
            color:#1f77b4;
            text-align:center;
            margin-top: 20px; /* 상단 마진을 줄여서 헤더를 위로 이동 */
        }
        .upload-area {
            border: 2px dashed #1f77b4;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 20px;
            margin-top: 50px; /* 좌측 구역의 상단 마진을 추가해서 아래로 이동 */
        }
        .description {
            margin-top: 50px; /* 설명 문구도 아래로 내리기 */
            font-size: 18px;
        }
        .service-info {
            margin-top: 40px; /* 우측 서비스 정보도 수평적으로 맞추기 위해 아래로 내림 */
        }
        .prediction-area {
            padding: 10px;
            text-align: center;
            font-size: 22px;
        }
        footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #333; /* 어두운 배경 */
            color: #ffffff; /* 밝은 글자 색상 */
            text-align: center;
            padding: 10px 0;
        }
        /* 전체 페이지 상단 여백을 줄여서 헤더를 위로 올림 */
        .block-container {
            padding-top: 0px;
        }

        /* 다크 모드에서도 푸터가 잘 보이도록 스타일 지정 */
        @media (prefers-color-scheme: dark) {
            footer {
                background-color: #000; /* 야간 모드에서는 더 어두운 배경 */
                color: #ffffff; /* 글자는 여전히 밝게 */
            }
        }

        @media (prefers-color-scheme: light) {
            footer {
                background-color: #f1f1f1; /* 라이트 모드 배경 */
                color: #000000; /* 라이트 모드 글자 색상 */
            }
        }
    </style>
""", unsafe_allow_html=True)

# 타이틀 표시
st.markdown('<h1 class="main-title">타이어 상태 분류 서비스 🚗</h1>', unsafe_allow_html=True)

# 레이아웃 구성 (좌측과 우측 사이에 여백 추가)
col1, spacer, col2 = st.columns([1, 0.2, 1])

# 좌측: 파일 업로드 및 예측 버튼
with col1:
    st.markdown('<div class="description">타이어 상태를 분석하고 결과를 빠르게 제공해드립니다.</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-area">이미지를 업로드하세요 (png, jpg, jpeg)</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # 업로드된 이미지 표시
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드한 이미지", use_column_width=True)

        # 예측 버튼 및 결과 표시
        if st.button("예측하기"):
            with st.spinner('예측 중입니다...'):
                try:
                    # Flask 서버로 이미지 전송
                    response = requests.post(
                        "http://172.31.15.63:5002/predict",
                        files={"file": uploaded_file.getvalue()}
                    )

                    # 예측 결과 표시
                    if response.status_code == 200:
                        prediction = response.json().get("predicted_class")
                        st.success(f"예측 결과: **{prediction}**")
                    else:
                        st.error(f"에러: {response.json().get('error')}")
                except Exception as e:
                    st.error(f"서버와 통신 중 에러 발생: {e}")
        else:
            st.info("예측 결과는 여기에 표시됩니다.")

# 우측: 서비스 설명 및 타이어 관련 이미지 추가
with col2:
    st.markdown('<div class="service-info">', unsafe_allow_html=True)  # 우측 구역에 마진 추가
    st.markdown("### 서비스 정보")
    st.write("""
        이 서비스는 업로드된 타이어 이미지를 분석하여 타이어 상태를 분류하는 AI 기반 서비스입니다.
        
        **사용 방법**:
        1. 좌측에 타이어 이미지를 업로드하세요.
        2. '예측하기' 버튼을 클릭하면, 서버에서 예측 결과를 받아옵니다.
        
        **모델 설명**:
        이 모델은 수천 개의 타이어 이미지 데이터를 학습하여, 손상된 타이어와 정상적인 타이어를 분류하는 방법을 배웠습니다.
        
        **예측 가능한 클래스**:
        - 정상 (Good)
        - 손상됨 (Defective)
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # 우측 하단: 만족도 조사 추가
    st.markdown("---")
    st.markdown("### 서비스 만족도 조사")
    st.write("서비스에 대한 만족도를 선택해 주세요:")

    # 만족도 조사 버튼
    col_like, col_dislike = st.columns([1, 1])
    with col_like:
        if st.button("👍 좋아요"):
            save_survey(True)  # 좋아요 데이터 저장
            st.success("감사합니다! 좋은 서비스를 제공할 수 있도록 노력하겠습니다.")
    with col_dislike:
        if st.button("👎 싫어요"):
            save_survey(False)  # 싫어요 데이터 저장
            st.warning("불편을 드려 죄송합니다. 더 나은 서비스를 위해 노력하겠습니다.")
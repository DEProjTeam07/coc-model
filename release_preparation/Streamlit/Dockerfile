# Python 3.11.9 Image 사용 
FROM python:3.11.9

# 컨테이너 내에서 작업할 디렉토리 설정
WORKDIR /app

# 필요한 패키지를 먼저 복사하고 설치한 후, requirements.txt 파일을 삭제
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

# Streamlit 로그 폴더 생성
RUN mkdir -p /app/log

# 포트 8502을 외부로 노출 
EXPOSE 8502
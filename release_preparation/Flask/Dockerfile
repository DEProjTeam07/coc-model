# Python 3.11.9 Image 사용 
FROM python:3.11.9

# 컨테이너 내에서 작업할 디렉토리 설정
WORKDIR /app

# 필요한 패키지를 먼저 복사하고 설치한 후, requirements.txt 파일을 삭제
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

# Flask도 설치 
RUN pip install Flask

# 여러 폴더를 한 번에 생성
RUN mkdir -p /app/log /app/src /app/templates

# 포트 5003을 외부로 노출 
EXPOSE 5003
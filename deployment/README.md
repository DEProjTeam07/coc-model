# 운영 서버 구성 
## 1. 개요
사용자가 웹 서비스를 사용하면서 데이터 과학자가 구현한 타이어 결함 모델을 통해 
본인이 업로드한 사진에 대해서 분석을 할 수 있다. 

사용자가 서비스에 대해서 좋아요, 싫어요와 같은 의견을 Online Metric으로 활용하여 추후 모델을 업데이트한다. 그렇게 함으로써 모델 성능을 높이고 서비스를 사용하는 사용자 만족도를 향상시키도록 한다.  

## 2. 운영 서버를 구성하는 4개 서비스 Preview 

- Streamlit 
  - 사용자가 인터넷 화면에 접속해서 보이는 클라이언트 화면을 의미한다. 
  - 예측하기 버튼을 클릭해서 타이어 이미지에 대한 추론을 할 수 있게끔 제공한다. 
  - 우측 하단에 좋아요, 싫어요 버튼을 클릭하여 DB에 서비스 만족도를 기록할 수 있도록 제공한다. 

- Flask 
  - 사용자가 입력한 이미지를 전처리해서 Mlflow Model Registry -> AWS S3에 저장되어 있는 모델 아티팩트를 다운로드해서 추론값을 Streamlit 웹 서비스에게 반환한다. 

- PostgreDB 
  - 좋아요, 싫어요 기록을 저장한다. 


- Grafana 
  - PostgreDB에 저장되어 있는 좋아요, 싫어요에 대한 시간대별 분석, 총 비율 분석, 추이 분석 등을 가능하게 하며, 데이터 상태에 대한 조건을 마련하여 Slack이나 Discord 같은 플랫폼에 알람을 전달할 수 있다. 


## 3. 4개 서비스에 대한 각각의 구현 내용 
  - Streamlit 
    - Dockerfile : Streamlit를 작동하기 위한 패키지를 설치한다. 그리고 로그 폴더를 생성한다. 이는 Streamlit 컨테이너에서 발생하는 로그를 저장하기 위함이다. 

    - streamlit_app.py : Streamlit 웹 서비스에 대한 구현 파이썬 파일 

    - log 폴더 : Streamlit 컨테이너에서 발생하는 로그를 로컬 호스트에서 쉽게 확인할 수 있도록 한다. 
    log 폴더에 streamlit_app.log 파일이 생길 것이다. 
    

  - Flask 
    - Dockerfile : Flask를 작동하기 위한 패키지를 설치한다. 그리고 로그 폴더를 생성한다. 이는 Flask 컨테이너에서 발생하는 로그를 저장하기 위함이다. 

    - flask_app.py : Flask 백엔드 서비스에 대한 구현 파이썬 파일 

    - communicate_mlflow.py : MLflow에서 Production Name의 모델의 uri를 가져오는 역할을 한다. 

    - log 폴더 : Flask 컨테이너에서 발생하는 로그를 로컬 호스트에서 쉽게 확인할 수 있도록 한다. log 폴더에 flaks_app.log 파일이 생길 것이다. 

  - PostgreDB 
    - crontab.py : 크론탭으로 DB에 랜덤으로 좋아요, 싫어요에 대한 데이터를 삽입하는 파일로 
    단기간에 데이터를 쌓기 위한 용도로 사용했을 뿐, 만약에 그럴 목적이 없다면 이 파이썬 파일은 필요하지 않다. 

    - init.sql : Postgre 컨테이너를 최초로 기동할 떄 테이블을 생성하는 sql

    - timezone.sql : Postgre 컨테이너를 최초로 기동할 떄 DB에 대한 Timezone 설정과 테이블의 속성에 대한 규칙 설정을 담당한다. 

  - Grafana : 일반적으로 docker-compose.yaml에서 도커 볼륨을 이용해서 Docker Hub에서 제공하는 Grafana Image를 불러와서 Container로 띄우는 방식으로 작업을 한다. 

<br>

## 4. 서비스를 Container로 띄우기 위한 docker-compose.yaml 설명 

```
 services:
  streamlit-app:
    build:
      context: ./Streamlit                                          # 루트 디렉터리를 ./Streamlit 폴더로 해놓는다. 
      dockerfile: Dockerfile                                        # ./Streamlit 내부의 Dockerfile 사용
    ports:
      - "${STREAMLIT_PORT}:8501"                                    # 호스트의 8501 포트를 컨테이너의 8501 포트로 매핑
    volumes:
      - ./Streamlit/streamlit_app.py:/app/streamlit_app.py          # 로컬의 streamlit_app.py를 컨테이너로 마운트
      - ./Streamlit/log:/app/log                                    # 로그 파일을 저장하기 위한 logs 디렉토리 마운트
    environment:                                                    # Streamlit에서 좋아요, 싫어요 버튼 클릭할 떄 DB와 연결하기 위한 환경 변수 설정                      
      POSTGRES_HOST: ${POSTGRES_HOST}                               # 같은 경로의 .env에서 자동 로드 
      POSTGRES_DB: ${POSTGRES_DB}                                   # 같은 경로의 .env에서 자동 로드 
      POSTGRES_PORT: ${POSTGRES_PORT}                               # 같은 경로의 .env에서 자동 로드 
      POSTGRES_USER: ${POSTGRES_USER}                               # 같은 경로의 .env에서 자동 로드 
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}                       # 같은 경로의 .env에서 자동 로드 
    command:                                                        # 사용자가 Streamlit 웹서비스에서 작동하다가 발생한 로그를 로그 파일로 리디렉션
      - /bin/bash
      - -c
      - |
        streamlit run /app/streamlit_app.py --server.address=0.0.0.0 >> /app/log/streamlit_app.log 2>&1    
    restart: always                                                # 컨테이너가 중지될 때 항상 다시 시작

  flask-app:
    build:
      context: ./Flask                                              # 루트 디렉터리를 ./Flask 폴더로 해놓는다. 
      dockerfile: Dockerfile                                        # ./Flask 폴더의 내부의 Dockerfile 사용
    ports:
      - "${FLASK_PORT}:5002"                                        # 호스트의 5002 포트를 컨테이너의 5002 포트로 매핑
    volumes:
      - ./Flask/flask_app.py:/app/flask_app.py                      # 로컬의 flask_app.py를 컨테이너로 마운트
      - ./Flask/communicate_mlflow.py:/app/communicate_mlflow.py    # 로컬에 있는 communicate_mlflow.py를 컨테이너로 마운트 
      - ./Flask/src/:/app/src/                                      # 로컬에 있는 CNN, EfficientNet과 같은 모델들을 컨테이너 경로에 배치 
      - ./Flask/templates/index.html:/app/templates/index.html                            # 로컬에 있는 index.html을 컨테이너 상 경로에 배치 
      - ./Flask/log:/app/log                                        # 로그 파일을 저장하기 위한 logs 디렉토리 마운트
    command:                                                        # Flask 실행 명령어와 로그 리디렉션
      - /bin/bash
      - -c
      - |
        python /app/flask_app.py >> /app/log/flask_app.log 2>&1
    restart: always    

  postgres:
    image: postgres:15
    ports:
      - "${POSTGRES_PORT}:5432"
    volumes:
      - db-data:/var/lib/postgresql/data  # PostgreSQL 데이터 저장
      - ./PostgreDB/init.sql:/docker-entrypoint-initdb.d/init.sql  # 초기 테이블 생성 SQL 스크립트
      - ./PostgreDB/timezone.sql:/docker-entrypoint-initdb.d/timezone.sql  # 타임존 설정 스크립트
    environment:
      POSTGRES_DB: ${POSTGRES_DB}                           # 같은 경로의 .env 파일에서 자동 로드 
      POSTGRES_USER: ${POSTGRES_USER}                       # 같은 경로의 .env 파일에서 자동 로드 
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}               # 같은 경로의 .env 파일에서 자동 로드 
    restart: always  

volumes:
  db-data:

```
 - 자세한 내용은 docker-compose.yaml에서 주석으로 설명해서 참고하면 된다. 
 - .env 파일은 Github 원격 레포지토리에 업로드하지 않는 것이 일반적이다. 
   docker-compose.yaml에서 환경 변수의 값으로 쓸 수 있었던 것은 같은 경로의 .env 파일을 생성해서 값을 정의했기 떄문이다. 


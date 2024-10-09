# crontab으로 DB 테이블에 좋아요, 싫어요 자동 기록하게 하기 
import psycopg2
from psycopg2 import sql
import random
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# PostgreSQL 연결 정보 설정 (환경 변수에서 불러오기)
db_config = {
    'host': os.getenv('POSTGRES_HOST'),
    'database': os.getenv('POSTGRES_DB'),
    'user': os.getenv('POSTGRES_USER'),
    'password': os.getenv('POSTGRES_PASSWORD'),
    'port': os.getenv('POSTGRES_PORT')
}

# 랜덤으로 True 또는 False 생성하는 함수
def get_random_is_good():
    return random.choice([True, False])

# 데이터베이스에 데이터 삽입하는 함수
def insert_data():
    try:
        # PostgreSQL 연결
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # 랜덤으로 is_good 값을 생성
        is_good = get_random_is_good()

        # 데이터 삽입 SQL 쿼리 작성
        insert_query = sql.SQL("""
            INSERT INTO service_survey (is_good)
            VALUES (%s);
        """)

        # 쿼리 실행
        cursor.execute(insert_query, (is_good,))

        # 변경 사항 저장
        conn.commit()

        print(f"Data inserted with is_good = {is_good}")

    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # 연결 종료
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    insert_data()
